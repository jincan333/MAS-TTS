import asyncio
from enum import Enum
from typing import Any, Dict, List, Tuple, Union
import copy
from colorama import Fore


from agentverse.environments import BaseEnvironment
from agentverse.agents.base import BaseAgent
from agentverse.logging import logger
from agentverse.utils import AGENT_TYPES
from agentverse.message import Message, SolverMessage, ExecutorMessage, CEOMessage


from .. import env_registry as EnvironmentRegistry

from agentverse.environments.tasksolving_env.rules import TasksolvingRule


@EnvironmentRegistry.register("task-basic")
class BasicEnvironment(BaseEnvironment):
    rule: TasksolvingRule
    agents: Dict[Enum, Union[BaseAgent, List[BaseAgent]]] = None

    task_description: str
    single_agent: bool = False
    cnt_turn: int = 0
    max_turn: int = 10
    success: bool = False
    ceo: bool = False

    def __init__(self, **kwargs):
        rule_config = kwargs.pop("rule", {})
        ceo_config = rule_config.pop("ceo", {"type": "none"})
        role_assigner_config = rule_config.pop(
            "role_assigner", {"type": "role_description"}
        )
        decision_maker_config = rule_config.pop("decision_maker", {"type": "vertical"})
        executor_config = rule_config.pop("executor", {"type": "none"})
        evaluator_config = rule_config.pop("evaluator", {"type": "basic"})
        rule = TasksolvingRule(
            ceo_config=ceo_config,
            role_assigner_config=role_assigner_config,
            decision_maker_config=decision_maker_config,
            executor_config=executor_config,
            evaluator_config=evaluator_config,
        )
        super().__init__(rule=rule, **kwargs)

    async def step(
        self, advice: str = "No advice yet.", previous_plan: str = "No solution yet.", current_resources: str = "No resources yet."
    ) -> List[Message]:
        result = ""
        logs = []

        logger.info(f"Loop Round {self.cnt_turn}")
        # ================== CEO ==================
        resources: List[CEOMessage] = await self.rule.resource_distribute(
            self.task_description, self.agents, previous_plan, advice, current_resources
        )
        critic_agent = copy.deepcopy(self.agents[AGENT_TYPES.CRITIC][0])
        current_resources = f"recruit_number: {resources['recruit_number']}, maximum_tokens: {resources['maximum_tokens']}"
        self.agents[AGENT_TYPES.CRITIC] = [copy.deepcopy(critic_agent) for _ in range(resources['recruit_number'] - 1)]

        logs.append({"module": "CEO", "content": resources})
        logger.info("", f"Resources:\n{resources}", Fore.CYAN)
        # ================== CEO ==================

        # ================== EXPERT RECRUITMENT ==================
        if not self.single_agent:
            agents = await self.rule.role_assign(
                self.task_description, self.agents, self.cnt_turn, advice
            )
            description = "\n".join([agent.role_description for agent in agents])
        else:
            description = "You are a scientist"
        self.agents[AGENT_TYPES.SOLVER].llm.args.max_tokens = resources['maximum_tokens']
        for temp_agent in self.agents[AGENT_TYPES.CRITIC]:
            temp_agent.llm.args.max_tokens = resources['maximum_tokens']
        advice = resources['direction']
        logs.append({"module": "Role Assigner", "content": description})
        logger.info("", f"Role Assignment:\n{description}", Fore.CYAN)
        # ================== EXPERT RECRUITMENT ==================

        # ================== DECISION MAKING ==================
        plan: List[SolverMessage] = await self.rule.decision_making(
            self.task_description, self.agents, previous_plan, advice
        )
        flatten_plan = "\n".join([p.content for p in plan])
        logs.append({"module": "Decision Maker", "content": flatten_plan})
        logger.info("", f"Decision Plan:\n{flatten_plan}", Fore.YELLOW)
        # ================== DECISION MAKING ==================

        # ================== EXECUTION ==================
        if not self.single_agent:
            result: List[ExecutorMessage] = await self.rule.execute(
                self.task_description, self.agents, plan
            )
            flatten_result = "\n".join([r.content for r in result])
        else:
            flatten_result = ""
        logs.append({"module": "Executor", "content": flatten_result})
        logger.info("", f"Execution Result:", Fore.GREEN)
        logger.info("", flatten_result, Fore.GREEN)
        # ================== EXECUTION ==================

        # ================== EVALUATION ==================
        if not self.single_agent:
            score, advice = await self.rule.evaluate(
                self.task_description, self.agents, plan, result
            )
        else:
            score = None
            advice = ""
        logs.append(
            {
                "agent": "evaluator",
                "content": f"Evaluation result: Score: {score}\nAdvice: {advice}",
            }
        )
        logger.info(
            "", f"Evaluation result:\nScore: {score}\nAdvice: {advice}", Fore.YELLOW
        )

        if score is not None and (
            (isinstance(score, bool) and score is True)
            or (isinstance(score, (list, tuple)) and all([s >= 8 for s in score]))
        ):
            # TODO: 8 is an arbitrary threshold
            logs.append({"agent": "system", "content": "Good score! Accept!"})
            logger.info(
                "", f"Good score! Accept! Final Result:\n{flatten_plan}", Fore.GREEN
            )
            self.success = True
        else:
            logs.append({"agent": "system", "content": "Bad score! Reject!"})
            logger.info("", "Bad score! Reject!", Fore.RED)
        self.cnt_turn += 1
        return flatten_result, advice, flatten_plan, current_resources, logs, self.success

    def iter_agents(self):
        for role, agent_or_agents in self.agents.items():
            if isinstance(agent_or_agents, list):
                for agent in agent_or_agents:
                    yield role, agent
            else:
                yield role, agent_or_agents

    def get_spend(self):
        total_spent = sum([agent.get_spend() for (_, agent) in self.iter_agents()])
        return total_spent

    def report_metrics(self) -> None:
        logger.info("", "Agent spend:", Fore.GREEN)
        for role, agent in self.iter_agents():
            name = agent.name.split(":")[0]
            logger.info(
                "",
                f"Agent (Role: {role}) {name}: {agent.get_spend_formatted()}",
                Fore.GREEN,
            )
        logger.info("", f"Total spent: ${self.get_spend():.6f}", Fore.GREEN)

    def is_done(self):
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turn or self.success

    def set_task_description(self, task_description: str = ""):
        self.task_description = task_description

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
