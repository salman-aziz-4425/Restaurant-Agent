from typing import Annotated
from pydantic import Field
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent

from src.agents.base_agent import RunContext_T

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
async def to_greeter(context: RunContext_T) -> Agent:
    """Return to greeter agent"""
    curr_agent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context) 