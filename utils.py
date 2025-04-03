from typing import Any, Union, Sequence, Callable

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from pydantic import BaseModel

# Utility function to wrap LLM with appropriate preprocessing
def wrap_agent_model(
        model: BaseChatModel, 
        messages_key: str="messages",
        instructions: str="", 
        output_format: Union[dict, BaseModel]="",
        tools: Sequence[Callable]=[], 
        **tool_kwargs,
    ) -> RunnableSerializable[Any, AIMessage]:
    # Wrap an LLM for agent, the agent can be equipped with SystemMessage from instructions, tools from tools, or restrict its output format by output.
    # A subsequent agent node will invoke this model inside the node.
    if len(instructions) > 0:
        preprocessor = RunnableLambda(
            lambda state: [SystemMessage(content=instructions)] + state[messages_key],
            name="StateModifier",
        )
    else:
        preprocessor = RunnableLambda(
            lambda state: state[messages_key],
            name="StateModifier",
        )
    if len(tools) > 0:
        model = model.bind_tools(tools, **tool_kwargs)
    if output_format:
        return preprocessor | model.with_structured_output(output_format)
    else:
        return preprocessor | model