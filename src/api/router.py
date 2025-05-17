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

from livekit.api import (
    AccessToken,
    RoomAgentDispatch,
    CreateAgentDispatchRequest,
    RoomConfiguration,
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


load_dotenv()

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "").strip()

if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        raise ValueError("LiveKit credentials not found in environment variables")

LIVEKIT_HOST = LIVEKIT_URL.replace("wss://", "").replace("ws://", "").replace("https://", "").replace("http://", "").strip()

WS_URL = f"wss://{LIVEKIT_HOST}"
API_URL = f"https://{LIVEKIT_HOST}"

app = FastAPI(title="Restaurant API")


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

# Initialize the worker pool
worker_pool = WorkerPool(max_workers=5)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    print("Room:")
    print(ctx.room)
    print("Room metadata:")
    print(ctx.room.metadata)
    print("Worker ID:")
    print(ctx.worker_id)
    print("Job ID:")
    print(ctx.job.id if ctx.job else "None")
    print("Job metadata:")
    
    job_metadata = {}
    if ctx.job and ctx.job.metadata:
        try:
            job_metadata = json.loads(ctx.job.metadata)
            print(f"Successfully parsed job metadata: {json.dumps(job_metadata, indent=2)}")
        except Exception as e:
            print(f"Failed to parse job metadata: {e}")
            print(f"Raw metadata: {ctx.job.metadata}")
    else:
        print("No job metadata available")

    menu = job_metadata.get("menu", MENU)
    print(f"Using menu: {menu}")


    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu,job_metadata.get("user_name", "Not specified")),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )

    print(f"Customer name from metadata: {job_metadata.get('user_name', 'Not specified')}")
    print(f"Session ID from metadata: {job_metadata.get('session_id', 'Not specified')}")

    try:
        session = AgentSession[UserData](
            userdata=userdata,
            stt=STT(model="nova-3", language="multi"),
            llm=LLM(model="gpt-4o-mini", timeout=30),
            tts=TTS(api_key=os.getenv("ELEVENLABS_API_KEY")),
            vad=VAD.load(),
        )

        

        print("Agent running and waiting for participants...")
        await session.start(
            room=ctx.room,
            agent=userdata.agents["greeter"],
            room_input_options=RoomInputOptions(),
        )
        print(f"Agent session started successfully")
        
        return session
    except Exception as e:
        print(f"Error in agent session: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/token")
@app.post("/token")
async def generate_token(request: Request, start_worker: bool = True, create_dispatch: bool = True):
    print("Generating token")
    if request.method == "GET":
        user_id = f"user-{uuid.uuid4()}"
        room_name = "restaurant-room"
        agent_name = "restaurant-agent"
    else:
        body = await request.json()
        user_id = body.get("user_id", f"user-{uuid.uuid4()}")
        room_name = body.get("room_name", "restaurant-room")
        agent_name = body.get("agent_name", "restaurant-agent")
        start_worker = body.get("start_worker", True)
        create_dispatch = body.get("create_dispatch", True)

    session_id = str(uuid.uuid4())
    unique_agent_name = f"{agent_name}-{session_id[:8]}"
    
    userdata = UserData()
    sessions[session_id] = userdata
    
    worker_metadata = {
        "session_id": session_id,
        "menu": MENU,
        "agent_name": unique_agent_name,
        "user_name":"Salman",
        "api_key": LIVEKIT_API_KEY,
        "api_secret": LIVEKIT_API_SECRET,
    }

    metadata = {
        "user_id": user_id,
        "session_id": session_id,
        "menu": MENU,
        "agent_type": "restaurant",
        "user_name":"Salman"
    }

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials not found in environment variables"
        )

    worker_pid = None
    worker_started = False
    worker_logs = None
    dispatch_result = None
    
    token = (
        AccessToken(api_key=api_key, api_secret=api_secret)
        .with_identity(user_id)
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .with_name(user_id)
        .with_metadata(json.dumps(metadata))
        .with_room_config(
            RoomConfiguration(
                agents=[
                    RoomAgentDispatch(agent_name=unique_agent_name, metadata=json.dumps(worker_metadata))
                ],
            ),
        )
        .to_jwt()
    )

    if start_worker:
        try:
            async def run_worker_in_background():
                try:
                    worker_id = f"{unique_agent_name}-{uuid.uuid4()}"
                    options = WorkerOptions(
                        entrypoint_fnc=entrypoint, 
                        agent_name=unique_agent_name,
                        ws_url=LIVEKIT_URL,
                        api_key=LIVEKIT_API_KEY,
                        api_secret=LIVEKIT_API_SECRET
                    )
                    
                    if await worker_pool.add_worker(worker_id, options):
                        try:
                            from livekit.agents.worker import Worker
                            worker = Worker(options, devmode=True, loop=asyncio.get_event_loop())
                            await worker.run()
                        finally:
                            await worker_pool.remove_worker(worker_id)
                    else:
                        print(f"Worker {worker_id} queued - pool is at capacity")
                except Exception as e:
                    print(f"Error in background worker: {e}")
                    import traceback
                    traceback.print_exc()
            
            asyncio.create_task(run_worker_in_background())
            
            worker_started = True
            await asyncio.sleep(2)
        except Exception as e:
            print(f"Error starting worker: {e}")
            import traceback
            traceback.print_exc()
    
    if True:
        print(f"Waiting for worker to register with LiveKit...")
        await asyncio.sleep(2)
        
        success = False
        max_dispatch_retries = 3
        dispatch_retry_count = 0
        
        while not success and dispatch_retry_count < max_dispatch_retries:
            try:
                if dispatch_retry_count > 0:
                    wait_time = 2 ** dispatch_retry_count
                    print(f"Retrying dispatch in {wait_time} seconds (attempt {dispatch_retry_count+1}/{max_dispatch_retries})...")
                    await asyncio.sleep(wait_time)
                
                print(f"Creating LiveKitAPI client with URL: {API_URL}")
                
                lkapi = api.LiveKitAPI(
                    url=API_URL,
                    api_key=api_key,
                    api_secret=api_secret
                )
     
                print(f"Creating dispatch for {unique_agent_name} in room {room_name} (attempt {dispatch_retry_count+1}/{max_dispatch_retries})...")
                    
                try:
                    print(f"Creating/updating room metadata for {room_name}")
                    room_metadata = {
                        "type": "restaurant",
                        "menu": MENU,
                        "created_at": datetime.datetime.now().isoformat(),
                        "user_name": metadata.get("user_name", ""),
                        "session_id": session_id
                    }
                    
                    try:
                        await lkapi.room.create_room(
                            api.CreateRoomRequest(
                                name=room_name,
                                metadata=json.dumps(room_metadata)
                            )
                        )
                        print(f"Created room {room_name} with metadata")
                    except Exception as e:
                        print(f"Updating existing room: {e}")
                        await lkapi.room.update_room(
                            api.UpdateRoomRequest(
                                room=room_name,
                                metadata=json.dumps(room_metadata)
                            )
                        )
                        print(f"Updated room {room_name} metadata")
                except Exception as e:
                    print(f"Failed to set room metadata: {e}")
                
                dispatch = await lkapi.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(
                        agent_name=unique_agent_name, 
                        room=room_name, 
                        metadata=json.dumps(worker_metadata)
                    )
                )
                print(dispatch)
                print(f"Listing dispatches for room {room_name}")
                dispatches = await lkapi.agent_dispatch.list_dispatch(room_name=room_name)
                
                dispatch_result = {
                    "created": True,
                    "agent_name": unique_agent_name,
                    "room": room_name,
                    "dispatch_id": str(dispatch.id) if dispatch.id else None,
                    "dispatch_count": len(dispatches),
                    "attempt": dispatch_retry_count + 1
                }
                print(f"Created dispatch for {unique_agent_name} in room {room_name}")
                success = True
            except Exception as e:
                dispatch_retry_count += 1
                dispatch_result = {
                    "created": False,
                    "error": str(e),
                    "attempt": dispatch_retry_count
                }
                print(f"Dispatch attempt {dispatch_retry_count}/{max_dispatch_retries} failed: {e}")
            finally:
                if 'lkapi' in locals():
                    await lkapi.aclose()

    response = {
        "token": token,
        "session_id": session_id,
        "room_name": room_name,
        "agent_name": unique_agent_name,
        "user_id": user_id,
        "host": LIVEKIT_HOST,
        "ws_url": WS_URL,
        "api_url": API_URL,
        "worker_started": worker_started,
        "worker_pid": worker_pid,
        "worker_logs": worker_logs,
        "dispatch": dispatch_result
    }
    
    return response

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

