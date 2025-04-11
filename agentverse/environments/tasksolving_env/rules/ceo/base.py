from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple, Any

from pydantic import BaseModel

from agentverse.agents import CEOAgent
from agentverse.message import SolverMessage, CEOMessage

from . import ceo_registry


class BaseCEO(BaseModel):
    """
    The base class of CEO.
    """

    def step(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> List[CEOMessage]:
        pass

    async def astep(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        current_resources: str,
        *args,
        **kwargs,
    ) -> List[CEOMessage]:
        pass

    def reset(self):
        pass


@ceo_registry.register("none")
class NoneCEO(BaseCEO):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [CEOMessage(content="")]
    
    async def astep(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [CEOMessage(content="")]

    def reset(self):
        pass


@ceo_registry.register("dummy")
class DummyCEO(BaseCEO):
    """
    The base class of execution.
    """

    def step(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [CEOMessage(content=s.content) for s in solution]
    
    async def astep(
        self,
        agent: CEOAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return [CEOMessage(content=s.content) for s in solution]

    def reset(self):
        pass
