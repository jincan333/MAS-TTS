from __future__ import annotations

import os
import subprocess
import multiprocessing
from typing import TYPE_CHECKING, Any, List, Tuple

from agentverse.logging import get_logger
from agentverse.agents import CEOAgent
from agentverse.message import CEOMessage, SolverMessage
from agentverse.logging import logger

from . import BaseCEO, ceo_registry


@ceo_registry.register("basic")
class BasicCEORule(BaseCEO):
    """
    Generates descriptions for each agent.
    """

    async def astep(
        self,
        agent: CEOAgent,
        task_description: str = "",
        solution: List[SolverMessage] = [],
        advice  : str = "",
        current_resources: str = "",
        *args,
        **kwargs,
    ) -> List[CEOMessage]:
        assert task_description != ""
        assert len(solution) > 0

        ceo_message = await agent.astep(task_description=task_description, solution=solution, advice=advice, current_resources=current_resources)
        resources = ceo_message.content

        return resources

    def reset(self):
        pass