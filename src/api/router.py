from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from livekit import api
import json
import os

import datetime
from dotenv import load_dotenv
import asyncio
from contextlib import asynccontextmanager
import logging

from livekit.api import (
    AccessToken,
    VideoGrants,
    LiveKitAPI,
)

from src.models.user_data import UserData
from src.agents.specialized_agents import Greeter, Reservation, Takeaway, Checkout
from livekit.agents import WorkerOptions, cli, JobContext, AgentSession
from livekit.plugins.openai import LLM
from livekit.plugins.elevenlabs import TTS
from livekit.plugins.deepgram import STT
from livekit.plugins.silero import VAD
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.agents.worker import Worker
from livekit import agents

load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "").strip()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL, ELEVEN_API_KEY]):
        raise ValueError("LiveKit credentials not found in environment variables")

# Ensure LIVEKIT_URL has proper protocol
if not LIVEKIT_URL.startswith(("wss://", "ws://", "https://", "http://")):
    LIVEKIT_URL = f"wss://{LIVEKIT_URL}"

LIVEKIT_HOST = LIVEKIT_URL.replace("wss://", "").replace("ws://", "").replace("https://", "").replace("http://", "").strip()

WS_URL = f"wss://{LIVEKIT_HOST}"
API_URL = f"https://{LIVEKIT_HOST}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up FastAPI application...")
    try:
        yield
        print("Shutting down FastAPI application...")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

app = FastAPI(title="Restaurant API", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, UserData] = {}

MENU = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"

class TokenRequest(BaseModel):
    user_id: str
    room_name: Optional[str] = "restaurant-room"
    agent_name: Optional[str] = "restaurant-agent"

class ReservationRequest(BaseModel):
    session_id: str
    time: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None

class OrderRequest(BaseModel):
    session_id: str
    items: List[str]

class PaymentRequest(BaseModel):
    session_id: str
    card_number: str
    expiry: str
    cvv: str

class SIPRequest(BaseModel):
    room_name: str
    phone_numbers: List[str]

class DispatchRule(BaseModel):
    room_name: str
    rule_name: str
    priority: int = 1
    target_participant: Optional[str] = None
    action: str  # "forward", "reject", "transfer"
    conditions: Dict[str, str]

class SIPInboundTrunkRequest(BaseModel):
    name: str
    phone_numbers: List[str]
    krisp_enabled: Optional[bool] = False
    inbound_numbers_geo_match: Optional[bool] = False
    inbound_numbers_area_code: Optional[str] = None

class RoomAgentDispatchRequest(BaseModel):
    agent_name: str
    metadata: Optional[str] = None

class RoomConfigurationRequest(BaseModel):
    agents: List[RoomAgentDispatchRequest]
    empty_timeout: Optional[int] = None
    max_participants: Optional[int] = None

class SIPDispatchRuleRequest(BaseModel):
    trunk_id: str
    room_prefix: str
    phone_numbers: List[str]
    name: str

def get_session(session_id: str) -> UserData:
    if session_id not in sessions:
        sessions[session_id] = UserData()
        sessions[session_id].agents.update({
            "greeter": Greeter(MENU),
            "reservation": Reservation(),
            "takeaway": Takeaway(MENU),
            "checkout": Checkout(MENU),
        })
    return sessions[session_id]

running_workers = {}

class WorkerPool:
    def __init__(self, max_workers=5):
        self.max_workers = max_workers
        self.active_workers = {}
        self.worker_queue = asyncio.Queue()
        self.worker_metrics = {}

    async def add_worker(self, worker_id: str, options: WorkerOptions):
        if len(self.active_workers) >= self.max_workers:
            await self.worker_queue.put((worker_id, options))
            return False
        
        self.active_workers[worker_id] = {
            "options": options,
            "start_time": datetime.datetime.now(),
            "status": "starting"
        }
        return True

    async def remove_worker(self, worker_id: str):
        if worker_id in self.active_workers:
            worker_info = self.active_workers.pop(worker_id)
            end_time = datetime.datetime.now()
            duration = (end_time - worker_info["start_time"]).total_seconds()
            
            if worker_id not in self.worker_metrics:
                self.worker_metrics[worker_id] = []
            self.worker_metrics[worker_id].append({
                "start_time": worker_info["start_time"],
                "end_time": end_time,
                "duration": duration
            })

            if not self.worker_queue.empty():
                next_worker_id, next_options = await self.worker_queue.get()
                await self.add_worker(next_worker_id, next_options)

    def get_worker_status(self, worker_id: str) -> dict:
        if worker_id in self.active_workers:
            return {
                "status": "active",
                "info": self.active_workers[worker_id]
            }
        return {
            "status": "inactive",
            "metrics": self.worker_metrics.get(worker_id, [])
        }

worker_pool = WorkerPool(max_workers=5)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def entrypoint(ctx: JobContext):
    logger.info("Starting entrypoint function")
    await ctx.connect()
    
    logger.info(f"Connected to room: {ctx.room.name if ctx.room else 'None'}")
    logger.debug(f"Room details: {ctx.room}")
    logger.debug(f"Room metadata: {ctx.room.metadata}")
    logger.info(f"Worker ID: {ctx.worker_id}")
    logger.info(f"Job ID: {ctx.job.id if ctx.job else 'None'}")
    
    try:
        logger.info("Initializing UserData and agents")
        userdata = UserData()
        userdata.agents.update({
            "greeter": Greeter(MENU),
            "reservation": Reservation(),
            "takeaway": Takeaway(MENU),
            "checkout": Checkout(MENU),
        })
        logger.debug("Agents initialized successfully")

        logger.info("Creating agent session")
        session = AgentSession[UserData](
            userdata=userdata,
            stt=STT(model="nova-3", language="multi"),
            llm=LLM(model="gpt-4o-mini", timeout=30),
            tts=TTS(api_key=os.getenv("ELEVEN_API_KEY")),
            vad=VAD.load(),
        )
        logger.debug("Agent session created successfully")

        logger.info("Starting agent session and waiting for participants...")
        await session.start(
            room=ctx.room,
            agent=userdata.agents["greeter"],
            room_input_options=RoomInputOptions(),
        )
        logger.info("Agent session started successfully")
        
        return session
    except Exception as e:
        logger.error(f"Error in agent session: {e}")
        logger.error("Full error details:", exc_info=True)
        raise

async def setup_worker():
    try:
     agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

    except Exception as e:
        logger.error(f"Error in worker setup: {e}")
        logger.info(f"API Key present: {bool(LIVEKIT_API_KEY)}")
        logger.info(f"API Secret present: {bool(LIVEKIT_API_SECRET)}")
        logger.info(f"WebSocket URL: {WS_URL}")
        logger.error("Full error details:", exc_info=True)
        raise

@app.get("/token")
@app.post("/token")
async def generate_token(request: Request):
    if request.method == "GET":
        user_id = f"user-{uuid.uuid4()}"
        room_name = "restaurant-room"
    else:
        body = await request.json()
        user_id = body.get("user_id", f"user-{uuid.uuid4()}")
        room_name = body.get("room_name", "restaurant-room")

    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        room_metadata = {
            "type": "restaurant",
            "menu": MENU,
            "created_at": datetime.datetime.now().isoformat(),
        }

        try:
            await livekit_api.room.create_room(
                api.CreateRoomRequest(
                    name=room_name,
                    metadata=json.dumps(room_metadata)
                )
            )
        except Exception:
            await livekit_api.room.update_room(
                api.UpdateRoomMetadataRequest(
                    room=room_name,
                    metadata=json.dumps(room_metadata)
                )
            )

        # Generate token
        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(user_id)
            .with_name(user_id)
            .with_grants(VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        return {
            "token": token,
            "room_name": room_name,
            "user_id": user_id,
            "host": LIVEKIT_HOST,
            "ws_url": WS_URL,
            "api_url": API_URL,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

@app.get("/workers/status")
async def get_workers_status():
    """Get status of all workers in the pool"""
    active_workers = {
        worker_id: {
            "status": "active",
            "start_time": info["start_time"].isoformat(),
            "uptime_seconds": (datetime.datetime.now() - info["start_time"]).total_seconds(),
            "agent_name": info["options"].agent_name
        }
        for worker_id, info in worker_pool.active_workers.items()
    }
    
    queue_size = worker_pool.worker_queue.qsize()
    
    metrics = {
        worker_id: {
            "total_jobs": len(metrics),
            "average_duration": sum(m["duration"] for m in metrics) / len(metrics) if metrics else 0,
            "last_active": max(m["end_time"] for m in metrics).isoformat() if metrics else None
        }
        for worker_id, metrics in worker_pool.worker_metrics.items()
    }
    
    return {
        "active_workers": active_workers,
        "queue_size": queue_size,
        "worker_metrics": metrics,
        "pool_capacity": worker_pool.max_workers
    }

@app.get("/worker/status")
async def get_worker_status():
    """Get the current status of the worker"""
    if not hasattr(app.state, 'worker_info'):
        return {"status": "not_initialized"}
    
    worker_info = app.state.worker_info
    task = worker_info["task"]
    
    status = {
        "id": worker_info["id"],
        "start_time": worker_info["start_time"].isoformat(),
        "uptime_seconds": (datetime.datetime.now() - worker_info["start_time"]).total_seconds(),
        "status": worker_info["status"],
        "is_running": not task.done() and not task.cancelled()
    }
    
    # Check if task has any exception
    if task.done():
        try:
            task.result()
        except Exception as e:
            status["error"] = str(e)
            status["status"] = "failed"
    
    return status

@app.get("/worker/health")
async def check_worker_health():
    """Quick health check endpoint for the worker"""
    if not hasattr(app.state, 'worker_info'):
        raise HTTPException(status_code=503, detail="Worker not initialized")
        
    task = app.state.worker_info["task"]
    if task.done() or task.cancelled():
        raise HTTPException(status_code=503, detail="Worker task not running")
        
    return {"status": "healthy"}

@app.post("/sip/connect")
async def create_sip_connection(request: SIPRequest):
    """Create a new SIP connection in a LiveKit room"""
    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        print(request.phone_numbers)
        trunk = api.SIPInboundTrunkInfo(
            name = request.room_name,
            numbers = request.phone_numbers,
            krisp_enabled = True,
        )

        request = api.CreateSIPInboundTrunkRequest(
            trunk = trunk
    )
        trunk = await livekit_api.sip.create_sip_inbound_trunk(request)
        print(trunk)
        return {
            "status": "success",
            "trunk_id": trunk.sip_trunk_id,
            "name": request.room_name,
            "inbound_numbers": request.phone_numbers,
            "krisp_enabled": True
        }
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

@app.post("/sip/dispatch-rule")
async def create_sip_dispatch_rule(request: SIPDispatchRuleRequest):
    """Create a new SIP dispatch rule"""
    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        rule = api.SIPDispatchRule(
            dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                room_prefix=request.room_prefix
            )
        )
        print(rule)
        dispatch_request = api.CreateSIPDispatchRuleRequest(
            rule=rule,
            trunk_ids=[request.trunk_id],
            inbound_numbers=request.phone_numbers,
            name=request.room_prefix
        )

        # Send the request
        response = await livekit_api.sip.create_sip_dispatch_rule(dispatch_request)
        
        print(response)
        logger.info(f"Created SIP dispatch rule: {response.sip_dispatch_rule_id}")
        
        return {
            "sip_dispatch_rule_id": response.sip_dispatch_rule_id,
            "rule": {
                "dispatch_rule_individual": {
                    "room_prefix": request.room_prefix
                }
            },
            "trunk_ids": request.trunk_id,
            "name": request.room_prefix,
            "inbound_numbers": request.phone_numbers[0] if request.phone_numbers else None
        }
    except Exception as e:
        logger.error(f"Error creating SIP dispatch rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

@app.get("/sip/dispatch-rules")
async def list_sip_dispatch_rules():
    """List all SIP dispatch rules"""
    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )

        response = await livekit_api.sip.list_sip_dispatch_rule(api.ListSIPDispatchRuleRequest())
        
        items = []
        for rule in response.items:
            items.append({
                "sip_dispatch_rule_id": str(rule.sip_dispatch_rule_id),
                "rule": {
                    "dispatch_rule_individual": {
                        "room_prefix": str(rule.rule.dispatch_rule_individual.room_prefix)
                    }
                },
                "trunk_ids": str(rule.trunk_ids),
                "inbound_numbers": str(rule.inbound_numbers),
                "room_config": {
                    "empty_timeout": int(rule.room_config.empty_timeout),
                    "max_participants": int(rule.room_config.max_participants),
                    "agents": [
                        {
                            "agent_name": str(agent.agent_name),
                            "metadata": str(agent.metadata)
                        }
                        for agent in rule.room_config.agents
                    ]
                }
            })
        
        return {
            "status": "success",
            "items": items
        }
    except Exception as e:
        logger.error(f"Error listing SIP dispatch rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

@app.get("/sip/inbound-trunk")
async def list_inbound_trunks():
    """List all SIP inbound trunks"""
    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )

        request = api.ListSIPInboundTrunkRequest()
        response = await livekit_api.sip.list_sip_inbound_trunk(request)
        
        # Convert response items to dictionary format
        items = []
        for trunk in response.items:
            items.append({
                "sip_trunk_id": str(trunk.sip_trunk_id),
                "name": str(trunk.name),
                "numbers": str(trunk.numbers),
                "krisp_enabled": bool(trunk.krisp_enabled)
            })
        
        return {
            "status": "success",
            "items": items
        }
    except Exception as e:
        logger.error(f"Error listing SIP trunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()

@app.delete("/sip/dispatch-rule/{sip_dispatch_rule_id}")
async def delete_sip_dispatch_rule(sip_dispatch_rule_id: str):
    """Delete a SIP dispatch rule"""
    try:
        livekit_api = LiveKitAPI(
            url=API_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )

        request = api.DeleteSIPDispatchRuleRequest(
            sip_dispatch_rule_id=sip_dispatch_rule_id
        )

        response = await livekit_api.sip.delete_sip_dispatch_rule(request)
        
        logger.info(f"Deleted SIP dispatch rule: {sip_dispatch_rule_id}")
        
        return {
            "status": "success",
            "message": f"Successfully deleted dispatch rule {sip_dispatch_rule_id}"
        }
    except Exception as e:
        logger.error(f"Error deleting SIP dispatch rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'livekit_api' in locals():
            await livekit_api.aclose()








