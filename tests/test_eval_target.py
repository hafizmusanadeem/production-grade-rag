from app.agents.nodes.decline import NO_CONTEXT_MESSAGE, OUT_OF_SCOPE_MESSAGE
from evals.target import AgentTarget, _make_state_with_history


class FakeAgent:
    def __init__(self,result=None,exc=None): self.result,self.exc,self.state=result,exc,None
    def invoke(self,state,config=None):
        self.state=state
        if self.exc: raise self.exc
        return self.result


def test_target_decline_flag(monkeypatch):
    import evals.target as m
    monkeypatch.setattr(m,"rag_agent",FakeAgent({"final_answer":NO_CONTEXT_MESSAGE,"declined":True}))
    assert AgentTarget().invoke("q").declined


def test_target_exact_fallback_decline(monkeypatch):
    import evals.target as m
    monkeypatch.setattr(m,"rag_agent",FakeAgent({"final_answer":OUT_OF_SCOPE_MESSAGE}))
    assert AgentTarget().invoke("q").declined


def test_history_query_is_last():
    state=_make_state_with_history("current",[{"role":"user","content":"old"},{"role":"assistant","content":"answer"}])
    assert state["messages"][-1]["content"]=="current"


def test_wall_clock_is_recorded(monkeypatch):
    import evals.target as m
    monkeypatch.setattr(m,"rag_agent",FakeAgent({"final_answer":"answer"}))
    assert AgentTarget().invoke("q").timings_ms["total"] > 0


def test_internal_error_is_returned_not_raised(monkeypatch):
    import evals.target as m
    monkeypatch.setattr(m,"rag_agent",FakeAgent(exc=RuntimeError("x")))
    assert AgentTarget().invoke("q").error=="internal_error"
