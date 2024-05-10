from typing import List, Dict, Annotated, Any, Optional, Sequence
from typing_extensions import TypedDict

import uuid
import json
import warnings
from datetime import datetime

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, AnyMessage, ToolCall
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnableConfig

from langgraph.utils import RunnableCallable
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import tools_condition, str_output
from langgraph.graph.message import add_messages

from database import Database
from policy import Policy
from online_search import PersianTavilySearchTool
from flight import FlightManager
from llm_translation import translate_to_persian


def get_tool_description(tool: BaseTool) -> str:
    tool_params = [
        f"{name}: {info['type']} ({info['description']})"
        for name, info in tool.args.items()
    ]
    tool_params_string = ', '.join(tool_params)
    return (
        f"tool_name -> {tool.name}\n"
        f"tool_params -> {tool_params_string}\n"
        f"tool_description ->\n{tool.description}"
    )


def get_tools_description(tools: List[BaseTool]) -> str:
    return '\n\n'.join([get_tool_description(tool) for tool in tools])


class ToolMessage(HumanMessage):
    """Ollama does not support `tool` role and the provided `ToolMessage` in Langchain"""

    def pretty_repr(self, html: bool = False) -> str:
        title = get_msg_title_repr("Tool" + " Message", bold=html)
        # TODO: handle non-string content.
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"


class ToolNode(RunnableCallable):

    def __init__(
        self,
        tools: Sequence[BaseTool],
        *,
        name: str = 'tools',
        tags: Optional[list[str]] = None,
    ) -> None:
        super().__init__(self._func, None, name=name, tags=tags, trace=False)
        self.tools_by_name = {tool.name: tool for tool in tools}

    def _func(
        self, input: dict[str, Any], config: RunnableConfig
    ) -> Any:
        message: AnyMessage = input['messages'][-1]
        if not isinstance(message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        def run_one(call: ToolCall):
            output = self.tools_by_name[call['name']].invoke(call['args'], config)
            tool_prompt = (
                "Here is the tool results:\n\n" +
                str_output(output)
            )
            return ToolMessage(
                content=tool_prompt, name=call['name'], tool_call_id=call['id']
            )

        return {'messages': run_one(message.tool_calls[0])}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state['messages'], config)
            try:
                content_json = json.loads(result.content)
                break
            except ValueError:
                warnings.warn('BAD FORMAT: ' + result.content)
                state['messages'] += [result, HumanMessage("Respond with a json output!")]

        action = content_json.get('ACTION', '').replace(' ', '')
        action_params = content_json.get('ACTION_PARAMS') or {}
        if type(action_params) is str:
            action_params = json.loads(action_params)
        final_answer = content_json.get('FINAL_ANSWER')

        if action:
            tool_call = ToolCall(name=action, args=action_params, id=str(uuid.uuid4()))
            result.tool_calls.append(tool_call)
            return {'messages': result}

        if not final_answer:
            persian_final_answer = ""
        else:
            persian_final_answer = translate_to_persian(final_answer, self.runnable)

        final_result = AIMessage(persian_final_answer)

        return {'messages': [result, final_result]}


SYSTEM_PROMPT_TEMPLATE = \
"""
You are a helpful Persian customer support assistant for Iran Airlines.
Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
When searching, be persistent. Expand your query bounds if the first search returns no results. 
If a search comes up empty, expand your search before giving up.

You have access to the following tools to get more information if needed:

{tool_descs}

You also have access to the history of privious messages.

Generate the response in the following json format:
{{
    "THOUGHT": "<you should always think about what to do>",
    "ACTION": "<the action to take, must be one tool_name from above tools>",
    "ACTION_PARAMS": "<the input parameters to the ACTION, it must be in json format complying with the tool_params>"
    "FINAL_ANSWER": "<a text containing the final answer to the original input question>",
}}
If you don't know the answer, you can take an action using one of the provided tools.
But if you do, don't take and action and leave the action-related attributes empty.
The values `ACTION` and `FINAL_ANSWER` can never ever be filled at the same time.
If you have any questions from the user, put that in `FINAL_ANSWER` as well.

Always make sure that your output is a json complying with above format.
Do NOT add anything before or after the json response.

Current user:\n<User>\n{user_info}\n</User>
Current time: {time}.
"""

class Agent:

    def __init__(self) -> None:
        self.database = Database(data_dir="storage/database")
        self.policy = Policy(data_dir="storage/policy", embedding=OllamaEmbeddings(model='llama3'))
        self.flight_manager = FlightManager(self.database)

        self.llm = ChatOllama(model='llama3', num_ctx=8192, num_thread=8, temperature=0.0)

        self._graph = self._build_graph()

    @property
    def tools(self) -> List[BaseTool]:
        return [
            PersianTavilySearchTool(max_results=3, llm=self.llm),
        ] + list(self.policy.get_tools().values()) + list(self.flight_manager.get_tools().values())

    def _build_graph(self) -> CompiledGraph:
        builder = StateGraph(State)

        builder.add_node('assistant', Assistant(self.llm))
        builder.add_node('action', ToolNode(self.tools))
        builder.set_entry_point('assistant')
        builder.add_conditional_edges(
            'assistant',
            tools_condition,
            {'action': 'action', END: END},
        )
        builder.add_edge('action', 'assistant')

        memory = SqliteSaver.from_conn_string(':memory:')
        graph = builder.compile(checkpointer=memory)
        return graph

    def _print_event(
        self, event: dict, printed_messages: set,
        ignore_first_system_message: bool = True,
    ) -> None:
        current_state = event.get('dialog_state')
        if current_state:
            print(f"Currently in: ", current_state[-1])

        messages = event.get('messages')
        if messages:
            if ignore_first_system_message:
                if isinstance(messages[0], SystemMessage):
                    messages = messages[1:]

            for message in messages:
                if message.id not in printed_messages:
                    msg_repr = message.pretty_repr(html=True)
                    print(msg_repr)
                    printed_messages.add(message.id)

    def run(
        self, question: str, config: Dict,
        reset_db: bool = True, clear_message_history: bool = True,
    ) -> None:
        if reset_db:
            self.database.reset_and_prepare()

        if clear_message_history:
            self._graph.checkpointer.put(config, checkpoint=empty_checkpoint())

        printed_messages = set()
        messages = []

        if clear_message_history:
            system_message = SystemMessage(SYSTEM_PROMPT_TEMPLATE.format(
                tool_descs=get_tools_description(self.tools),
                time=datetime.now(),
                user_info=f"passenger_id: {config.get('configurable', {}).get('passenger_id', None)}"
            ))
            messages.append(system_message)

        user_message = HumanMessage(question)
        messages.append(user_message)

        events = self._graph.stream(
            {'messages': messages}, config, stream_mode='values'
        )

        for event in events:
            self._print_event(event, printed_messages)
