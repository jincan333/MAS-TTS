from __future__ import annotations

import asyncio
from colorama import Fore

from agentverse.logging import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Tuple

# from agentverse.environments import PipelineEnvironment
from agentverse.message import SolverMessage, Message, CriticMessage
from agentverse.llms.base import LLMResult
from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent
from agentverse.utils import AgentCriticism
from agentverse.llms.utils.token_counter import count_string_tokens
from agentverse.llms.openai import invert_args, revert_args
from agentverse.llms import LOCAL_LLMS
logger = get_logger()


@agent_registry.register("solver")
class SolverAgent(BaseAgent):
    max_history: int = 5

    def step(
        self, former_solution: str, advice: str, task_description: str = "", **kwargs
    ) -> SolverMessage:
        pass

    async def astep(
        self, former_solution: str, advice: str, task_description: str = "", **kwargs
    ) -> SolverMessage:
        """Asynchronous version of step"""
        logger.debug("", self.name, Fore.MAGENTA)
        # prompt = self._fill_prompt_template(
        #     former_solution, critic_opinions, advice, task_description
        # )
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            former_solution=former_solution,
            task_description=task_description,
            advice=advice,
            role_description=self.role_description,
            **kwargs,
        )

        max_send_token = self.llm.send_token_limit(self.llm.args.model)
        max_send_token -= prompt_token

        history = await self.memory.to_messages(
            self.name,
            start_index=-self.max_history,
            max_send_token=max_send_token,
            model=self.llm.args.model,
        )
        if hasattr(self.llm.args, 'max_tokens'):
            llm_config = {
                "presence_penalty": self.llm.args.presence_penalty,
                "frequency_penalty": self.llm.args.frequency_penalty,
            "top_p": self.llm.args.top_p,
            "temperature": self.llm.args.temperature,
            "max_tokens": self.llm.args.max_tokens,
            }
        else:
            llm_config = None
        parsed_response = None
        for i in range(self.max_retry):
            try:
                if self.llm.args.model in LOCAL_LLMS:
                # if self.llm.args.model == "d1-32b":
                    response = self.llm.agenerate_response_local(
                        prepend_prompt, history, append_prompt
                    )
                else:
                    response = await self.llm.agenerate_response(
                        prepend_prompt, history, append_prompt
                    )
                # if count_string_tokens(response.content, self.llm.args.model) >= 20000 and self.llm.args.model in LOCAL_LLMS:
                #     logger.warn("Solver Retrying...")
                #     invert_args(self.llm.args, llm_config)
                #     continue
                parsed_response = self.output_parser.parse(response)
                if parsed_response.return_values["output"] == "":
                    logger.warn("Empty response from Solver, Solver Retrying...")
                    invert_args(self.llm.args, llm_config)
                    continue
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.error(e)
                logger.warn("Solver Retrying...")
                invert_args(self.llm.args, llm_config)
                continue
        revert_args(self.llm.args, llm_config)
        if parsed_response is None or parsed_response.return_values["output"] == "":
            logger.error(f"{self.name} failed to generate valid response.")
            response = "```I don't know.```"
            parsed_response = self.output_parser.parse(LLMResult(content=response))

        message = SolverMessage(
            content=parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    def _fill_prompt_template(
        self,
        former_solution: str,
        critic_opinions: List[AgentCriticism],
        advice: str,
        task_description: str,
    ) -> str:
        """Fill the placeholders in the prompt template

        In the role_assigner agent, three placeholders are supported:
        - ${task_description}
        - ${former_solution}
        - ${critic_messages}
        - ${advice}
        """
        input_arguments = {
            "task_description": task_description,
            "former_solution": former_solution,
            "critic_opinions": "\n".join(
                [
                    f"{critic.sender_agent.role_description} said: {critic.criticism}"
                    for critic in critic_opinions
                ]
            ),
            "advice": advice,
        }
        # if discussion_mode:
        #     template = Template(self.prompt_template[1])
        # else:
        template = Template(self.prompt_template)
        return template.safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
