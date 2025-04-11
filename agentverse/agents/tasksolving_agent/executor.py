from __future__ import annotations

from agentverse.logging import get_logger
from colorama import Fore
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Any

from agentverse.message import ExecutorMessage, Message, SolverMessage
from agentverse.utils import AgentFinish, AgentAction
from agentverse.llms.utils.token_counter import count_string_tokens
from agentverse.llms.openai import invert_args, revert_args
from agentverse.llms import LOCAL_LLMS
from agentverse.agents import agent_registry
from agentverse.agents.base import BaseAgent
import requests

logger = get_logger()


@agent_registry.register("executor")
class ExecutorAgent(BaseAgent):
    max_history: int = 5

    def step(
        self, task_description: str, solution: str, tools: List[dict] = [], **kwargs
    ) -> ExecutorMessage:
        pass

    async def astep(
        self, task_description: str, solution: str, tools: List[dict] = [], **kwargs
    ) -> ExecutorMessage:
        logger.debug("", self.name, Fore.MAGENTA)
        prepend_prompt, append_prompt, prompt_token = self.get_all_prompts(
            task_description=task_description,
            solution=solution,
            agent_name=self.name,
            **{**kwargs, "model": self.llm.args.model},
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
                        prepend_prompt, history, append_prompt, tools
                    )
                else:
                    response = await self.llm.agenerate_response(
                        prepend_prompt, history, append_prompt, tools
                    )
                # if count_string_tokens(response.content, self.llm.args.model) >= 20000 and self.llm.args.model in LOCAL_LLMS:
                #     logger.warn("Executor Retrying...")
                #     invert_args(self.llm.args, llm_config)
                #     continue
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.error(e)
                logger.warn("Executor Retrying...")
                continue

        revert_args(self.llm.args, llm_config)
        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")
            parsed_response = AgentAction(tool="", tool_input="", log={"file_path": f"test-{self.llm.args.model}/nonvalid.py", "code": "", "command": ""})
        if isinstance(parsed_response, AgentFinish):
            message = ExecutorMessage(
                content=parsed_response.return_values["output"],
                sender=self.name,
                sender_agent=self,
            )
        elif isinstance(parsed_response, AgentAction):
            message = ExecutorMessage(
                content=parsed_response.log,
                sender=self.name,
                sender_agent=self,
                tool_name=parsed_response.tool,
                tool_input=parsed_response.tool_input,
            )
        else:
            raise ValueError(
                f"Error response type: {type(parsed_response)}. Only support \
                    AgentFinish and AgentAction. Modify your output parser."
            )
        return message

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
