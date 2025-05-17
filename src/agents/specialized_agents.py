import logging
import os
from dataclasses import dataclass
from typing import Annotated, Optional, Tuple, Dict, ClassVar
from functools import lru_cache

from pydantic import Field

from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, RunContext
from livekit.plugins import deepgram, openai, silero, elevenlabs

from src.models import UserData

logger = logging.getLogger("restaurant-agents")

class VoiceConfig:
    VOICE_IDS: ClassVar[Dict[str, str]] = {
        "greeter": "21m00Tcm4TlvDq8ikWAM",
        "reservation": "ErXwobaYiN019PkySvjV",
        "takeaway": "2EiwWnXFnvU5JabPnv8n",
        "checkout": "TX3LPaxmHKxFdv7VOQHJ"
    }

    @classmethod
    def get_voice(cls, agent_type: str) -> str:
        return cls.VOICE_IDS[agent_type]

RunContext_T = RunContext[UserData]

@lru_cache(maxsize=128)
def get_cached_context(agent_name: str, user_data: str) -> str:
    return f"You are {agent_name} agent. Current user data is {user_data}"

@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"

@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"

@function_tool()
async def to_greeter(context: RunContext_T) -> Tuple[Agent, str]:
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)

class BaseAgent(Agent):
    _chat_ctx_cache = {}
    _transfer_messages = {
        "reservation": "I'll transfer you to our reservation agent who will help you book a table.",
        "takeaway": "I'll connect you with our takeaway agent who will help you place your order.",
        "checkout": "I'll transfer you to our checkout agent to process your payment.",
        "greeter": "I'll transfer you back to our greeter who can help you with anything else."
    }

    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        cache_key = (agent_name, id(userdata.prev_agent))
        if cache_key in self._chat_ctx_cache:
            chat_ctx = self._chat_ctx_cache[cache_key]
        else:
            if isinstance(userdata.prev_agent, Agent):
                truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                    exclude_instructions=True, exclude_function_call=False
                ).truncate(max_items=6)
                existing_ids = {item.id for item in chat_ctx.items}
                items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
                chat_ctx.items.extend(items_copy)
            
            system_message = get_cached_context(agent_name, userdata.summarize())
            chat_ctx.add_message(role="system", content=system_message)
            self._chat_ctx_cache[cache_key] = chat_ctx

        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent
        
        transfer_message = self._transfer_messages.get(name, f"Transferring to {name}.")
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(role="assistant", content=transfer_message)
        await self.update_chat_ctx(chat_ctx)
        
        return next_agent, transfer_message

class Greeter(BaseAgent):
    def __init__(self, menu: str, user_name: str = "") -> None:
        super().__init__(
            instructions=f"""You are a friendly restaurant greeter. 
                The menu is: {menu}
                If you know the customer's name, use it.
                Current customer name: {user_name}
                You can transfer to reservation, takeaway, or checkout agent.
                Before transferring, you MUST tell the customer which agent you are transferring to and why.
                For example:
                - For reservations: "I'll transfer you to our reservation agent who will help you book a table."
                - For takeaway: "I'll connect you with our takeaway agent who will help you place your order."
                - For checkout: "I'll transfer you to our checkout agent to process your payment."
                Be concise but welcoming.""",
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            stt=deepgram.STT(),
            tts=elevenlabs.TTS(voice_id=VoiceConfig.get_voice("greeter")),
            vad=silero.VAD.load(),
        )

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)
    
    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("checkout", context)

class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a restaurant reservation agent.
                Help customers book tables and manage their reservations.
                Be efficient and professional.
                Ask for reservation time, then customer's name, and phone number.
                Confirm all reservation details with the customer.""",
            llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
            stt=deepgram.STT(),
            tts=elevenlabs.TTS(voice_id=VoiceConfig.get_voice("reservation")),
            vad=silero.VAD.load(),
            tools=[update_name, update_phone, to_greeter],
            allow_interruptions=True,
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | Tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."
        if not userdata.reservation_time:
            return "Please provide reservation time first."
        return await self._transfer_to_agent("greeter", context)

class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=f"""You are a takeaway order agent.
                The menu is: {menu}
                Help customers place orders for pickup.
                Be efficient and accurate.
                Guide customers through menu selection and confirm their order.""",
            llm=openai.LLM(model="gpt-4", temperature=0.7),
            stt=deepgram.STT(),
            tts=elevenlabs.TTS(voice_id=VoiceConfig.get_voice("takeaway")),
            vad=silero.VAD.load(),
            tools=[to_greeter],
            allow_interruptions=True,
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | Tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."
        return await self._transfer_to_agent("checkout", context)

class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=f"""You are a checkout agent.
                The menu is: {menu}
                Help customers complete their orders and process payments.
                Be professional and thorough.
                Collect payment details securely and confirm all transaction details.""",
            llm=openai.LLM(model="gpt-4", temperature=0.7),
            stt=deepgram.STT(),
            tts=elevenlabs.TTS(voice_id=VoiceConfig.get_voice("checkout")),
            vad=silero.VAD.load(),
            tools=[update_name, update_phone, to_greeter],
             allow_interruptions=True,
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | Tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."
        if not all([
            userdata.customer_credit_card,
            userdata.customer_credit_card_expiry,
            userdata.customer_credit_card_cvv
        ]):
            return "Please provide the credit card information first."
        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context) 