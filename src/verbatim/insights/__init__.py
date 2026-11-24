"""Step 5: the insights agent on Google ADK, kept apart from the LangGraph version.

- agent.py: an ADK LlmAgent with the shared analyst tools, a guardrail callback,
  and a runner for one question; deploy/deploy_agent.py ships it to Agent Runtime
"""

from verbatim.insights.agent import APP_NAME, ask_insights_agent, build_insights_agent

__all__ = ["APP_NAME", "ask_insights_agent", "build_insights_agent"]
