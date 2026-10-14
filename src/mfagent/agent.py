"""The LangGraph agent.

    START -> agent --(asked for tools)--> tools -> agent -> ... -> END
                   \--(plain answer)-----> END

The agent node calls the LLM. If the LLM asks for tools, the tools node runs
them and their results go back to the LLM, which can ask for more tools or
answer. MAX_TOOL_ROUNDS stops a model that keeps asking for tools forever.
"""

from datetime import date
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from mfagent.client import FundApiError, FundSource
from mfagent.tools import build_tools

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are a research assistant for Indian mutual funds. Today is {today}.

Rules:
- Every number you state must come from a tool result in this conversation. Never
  estimate, recall or calculate a figure yourself.
- Category names are inconsistent, so find the exact names with list_categories before
  calling search_funds. A category such as Large Cap can exist under two spellings;
  search both and say so.
- Fields ending in _pct are already percentages. Mention the data's as_of_date.
- A field missing from a result means the data doesn't have it (the history only goes
  back about two years, so 3- and 5-year figures are usually missing). Say that plainly
  instead of guessing.
- If a tool returns an error, explain it to the user; don't retry the same call.
- Describe and compare; don't tell the user what to buy. End with one line noting that
  past returns don't guarantee future ones.
"""


class AgentState(MessagesState):
    tool_rounds: int


def build_agent(llm: BaseChatModel, source: FundSource):
    tools = build_tools(source)
    model = llm.bind_tools(tools)
    system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat()))

    def agent(state: AgentState) -> dict:
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(
                "I couldn't finish this within the allowed number of lookups. "
                "Try a narrower question, e.g. one category or one fund.")]}
        return {"messages": [model.invoke([system, *state["messages"]])]}

    def count_round(state: AgentState) -> dict:
        return {"tool_rounds": state.get("tool_rounds", 0) + 1}

    def route(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    # Pipeline errors (unknown fund, API down) go back to the model as the tool's
    # result, so it can explain them instead of the whole run crashing.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=(FundApiError,)))
    graph.add_node("count_round", count_round)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "count_round")
    graph.add_edge("count_round", "agent")
    return graph.compile()
