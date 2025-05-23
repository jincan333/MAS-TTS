from __future__ import annotations

import asyncio
from colorama import Fore

from agentverse.logging import get_logger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Tuple

from agentverse.message import EvaluatorMessage, Message
from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent
from agentverse.llms.utils.token_counter import count_string_tokens
from agentverse.llms.openai import invert_args, revert_args
from agentverse.llms import LOCAL_LLMS

logger = get_logger()


@agent_registry.register("evaluator")
class EvaluatorAgent(BaseAgent):
    max_history: int = 5

    def step(
        self,
        solution: str,
        result: str,
        task_description: str,
        all_role_description: str,
    ) -> EvaluatorMessage:
        pass
        # return parsed_response

    async def astep(
        self,
        solution: str,
        result: str,
        task_description: str,
        all_role_description: str,
    ) -> EvaluatorMessage:
        """Asynchronous version of step"""
        logger.debug("", self.name, Fore.MAGENTA)
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            solution=solution,
            result=result,
            task_description=task_description,
            all_role_description=all_role_description,
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
                #     logger.warn("Evaluator Retrying...")
                #     invert_args(self.llm.args, llm_config)
                #     continue
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.error(e)
                logger.warn("Evaluator Retrying...")
                invert_args(self.llm.args, llm_config)
                continue
        
        revert_args(self.llm.args, llm_config)
        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")
            parsed_response = (0, "")
        message = EvaluatorMessage(
            sender=self.name,
            sender_agent=self,
            score=parsed_response[0],
            advice=parsed_response[1],
        )
        return message

    def _fill_prompt_template(self, solution: str, task_description: str) -> str:
        """Fill the placeholders in the prompt template

        In the role_assigner agent, three placeholders are supported:
        - ${task_description}
        - ${solution}
        """
        input_arguments = {
            "task_description": task_description,
            "solution": solution,
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
