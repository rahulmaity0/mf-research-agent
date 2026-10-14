from conftest import FakeSource, scorecard
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from mfagent.agent import MAX_TOOL_ROUNDS, build_agent


class ScriptedModel(BaseChatModel):
    """Stands in for the LLM: returns the given replies in order and records what it was sent."""

    replies: list[AIMessage]
    received: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def call(name, args, id="1"):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": id}])


def run(model, source, question="q"):
    return build_agent(model, source).invoke({"messages": [HumanMessage(question)], "tool_rounds": 0})


def test_tool_result_goes_back_to_the_model_before_it_answers():
    model = ScriptedModel(replies=[call("get_fund_details", {"scheme_code": 7}), AIMessage("Fund 7 returned 10%.")])

    result = run(model, FakeSource([scorecard(7)]))

    assert result["messages"][-1].content == "Fund 7 returned 10%."
    tool_reply = model.received[1][-1]
    assert isinstance(tool_reply, ToolMessage)
    assert '"ret_1y_pct": 10.0' in tool_reply.content


def test_plain_answer_ends_without_calling_tools():
    model = ScriptedModel(replies=[AIMessage("Hello")])

    result = run(model, FakeSource())

    assert result["tool_rounds"] == 0
    assert len(model.received) == 1


def test_unknown_fund_is_reported_to_the_model_instead_of_crashing():
    model = ScriptedModel(replies=[call("get_fund_details", {"scheme_code": 999}), AIMessage("No such fund.")])

    result = run(model, FakeSource())

    tool_reply = model.received[1][-1]
    assert tool_reply.status == "error"
    assert "scheme 999 not found" in tool_reply.content
    assert result["messages"][-1].content == "No such fund."


def test_model_that_keeps_calling_tools_is_stopped():
    replies = [call("list_categories", {}, id=str(i)) for i in range(MAX_TOOL_ROUNDS + 5)]
    model = ScriptedModel(replies=replies)

    result = run(model, FakeSource())

    assert len(model.received) == MAX_TOOL_ROUNDS
    assert result["tool_rounds"] == MAX_TOOL_ROUNDS
    assert "allowed number of lookups" in result["messages"][-1].content


def test_system_prompt_is_sent_first():
    model = ScriptedModel(replies=[AIMessage("ok")])

    run(model, FakeSource())

    assert "Every number you state must come from a tool result" in model.received[0][0].content
