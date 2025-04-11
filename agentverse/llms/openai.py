import logging
import json
import ast
import os
import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union, ClassVar
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import Field
import re
import time
from transformers import AutoTokenizer
import asyncio

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message
from agentverse.llms.utils.token_counter import count_string_tokens

from . import llm_registry, LOCAL_LLMS, LOCAL_LLMS_MAPPING
from .base import BaseChatModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair
from .utils.llm_server_utils import get_llm_server_modelname


def extract_reasoner(response):
    match = re.search(r'<Final Answer>:(.*)', response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        response = content.strip()
    matches = list(re.finditer(r'</think>', response, re.DOTALL))
    if matches:
        last_match = matches[-1]
        content = response[last_match.end():].strip()
        response = content
    return response

def invert_args(args, config):
    if hasattr(args, 'max_tokens'):
        args.presence_penalty += 0.01 if args.presence_penalty < 2 else 0.0
        args.frequency_penalty += 0.02 if args.frequency_penalty < 2 else 0.0
        args.top_p -= 0.01 if args.top_p > 0.1 else 0.0
        args.temperature -= 0.01 if args.temperature > 0.1 else 0.0
        args.presence_penalty = min(args.presence_penalty, 2.0)
        args.frequency_penalty = min(args.frequency_penalty, 2.0)
        args.max_tokens = int(args.max_tokens * 0.5) if args.max_tokens > 8192 else args.max_tokens
    return args

def revert_args(args, config):
    if hasattr(args, 'max_tokens'):
        args.presence_penalty = config['presence_penalty']
        args.frequency_penalty = config['frequency_penalty']
        args.top_p = config['top_p']
        args.temperature = config['temperature']
        args.max_tokens = config['max_tokens']
    return args

try:
    from openai import OpenAI, AsyncOpenAI
    from openai import OpenAIError
    from openai import AzureOpenAI, AsyncAzureOpenAI
except ImportError:
    is_openai_available = False
    logger.warn(
        "openai package is not installed. Please install it via `pip install openai`"
    )
else:
    api_key = None
    base_url = None
    model_name = None
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
    AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE")
    DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_TOKEN")
    DEEPINFRA_BASE_URL = os.environ.get("DEEPINFRA_BASE_URL")
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL")
    TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
    TOGETHER_BASE_URL = os.environ.get("TOGETHER_BASE_URL")
    VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
    VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

    if not OPENAI_API_KEY and not AZURE_API_KEY:
        logger.warn(
            "OpenAI API key is not set. Please set an environment variable OPENAI_API_KEY or "
            "AZURE_OPENAI_API_KEY."
        )
    elif OPENAI_API_KEY:
        DEFAULT_CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        DEFAULT_CLIENT_ASYNC = AsyncOpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
        )
        api_key = OPENAI_API_KEY
        base_url = OPENAI_BASE_URL
    elif AZURE_API_KEY:
        DEFAULT_CLIENT = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
        DEFAULT_CLIENT_ASYNC = AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
        )
        api_key = AZURE_API_KEY
        base_url = AZURE_API_BASE
    if VLLM_BASE_URL:
        if model_name := get_llm_server_modelname(VLLM_BASE_URL, VLLM_API_KEY, logger):
            # model_name = /mnt/llama/hf_models/TheBloke_Llama-2-70B-Chat-GPTQ
            # transform to TheBloke/Llama-2-70B-Chat-GPTQ
            hf_model_name = model_name.split("/")[-1].replace("_", "/")
            LOCAL_LLMS.append(model_name)
            LOCAL_LLMS_MAPPING[model_name] = {
                "hf_model_name": hf_model_name,
                "base_url": VLLM_BASE_URL,
                "api_key": VLLM_API_KEY if VLLM_API_KEY else "EMPTY",
            }
            logger.info(f"Using vLLM model: {hf_model_name}")
    if hf_model_name := get_llm_server_modelname(
        "http://localhost:5000", logger=logger
    ):
        # meta-llama/Llama-2-7b-chat-hf
        # transform to llama-2-7b-chat-hf
        short_model_name = model_name.split("/")[-1].lower()
        LOCAL_LLMS.append(short_model_name)
        LOCAL_LLMS_MAPPING[short_model_name] = {
            "hf_model_name": hf_model_name,
            "base_url": "http://localhost:5000/v1",
            "api_key": "EMPTY",
        }
        logger.info(f"Using FSChat model: {model_name}")


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: float = Field(default=0)
    frequency_penalty: float = Field(default=0)
    max_completion_tokens: int = Field(default=32768)


# To support your own local LLMs, register it here and add it into LOCAL_LLMS.
@llm_registry.register("gpt-35-turbo")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
@llm_registry.register("vllm")
@llm_registry.register("local")
@llm_registry.register("gpt-4o")
@llm_registry.register("gpt-4o-mini")
@llm_registry.register("o3-mini")
@llm_registry.register("deepseek-chat")
@llm_registry.register("deepseek-reasoner")
@llm_registry.register("qwen2.5-32b-instruct")
@llm_registry.register("s1.1-32b")
@llm_registry.register("deepseek-r1-distill-qwen-32b")
@llm_registry.register("d1-32b")
@llm_registry.register("m1-32b")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)
    client_args: Optional[Dict] = Field(
        default={"api_key": api_key, "base_url": base_url}
    )
    is_azure: bool = Field(default=False)
    dataset: str = Field(default="")
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    model_cache: ClassVar[dict] = {}

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()
        client_args = {"api_key": api_key, "base_url": base_url}
        # check if api_key is an azure key
        is_azure = False
        dataset = kwargs.pop("dataset", "")
        if AZURE_API_KEY and not OPENAI_API_KEY:
            is_azure = True
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.warn(f"Unused arguments: {kwargs}")
        if args["model"] in LOCAL_LLMS:
            if args["model"] in LOCAL_LLMS_MAPPING:
                client_args["api_key"] = LOCAL_LLMS_MAPPING[args["model"]]["api_key"]
                client_args["base_url"] = LOCAL_LLMS_MAPPING[args["model"]]["base_url"]
                is_azure = False
            else:
                raise ValueError(
                    f"Model {args['model']} not found in LOCAL_LLMS_MAPPING"
                )
        super().__init__(
            args=args, max_retry=max_retry, client_args=client_args, is_azure=is_azure, dataset=dataset
        )

    def invert(self):
        if self.args.model == "o3-mini" and hasattr(self.args, 'max_tokens'):
            del self.args.max_tokens
            del self.args.temperature
            del self.args.top_p
        elif hasattr(self.args, 'max_completion_tokens'):
            del self.args.max_completion_tokens

        if self.args.model in ('deepseek-chat', 'deepseek-reasoner'):
            self.args.model = "deepseek-ai/DeepSeek-V3" if self.args.model == 'deepseek-chat' else "deepseek-ai/DeepSeek-R1"
        elif self.args.model in LOCAL_LLMS:
            self.args.model = LOCAL_LLMS_MAPPING[self.args.model]["hf_model_name"]

    def revert(self):
        if self.args.model in ('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-R1'):
            self.args.model = "deepseek-chat" if self.args.model == "deepseek-ai/DeepSeek-V3" else "deepseek-reasoner"
        elif self.args.model in LOCAL_LLMS:
            self.args.model = self.args.model.split('/')[-1].lower()
            if len(self.args.model) > 40:
                self.args.model = "m1-32b"

    @classmethod
    def send_token_limit(self, model: str) -> int:
        send_token_limit_dict = {
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0613": 16384,
            "gpt-3.5-turbo-1106": 16384,
            "gpt-3.5-turbo-0125": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-0613": 32768,
            "gpt-4-1106-preview": 131072,
            "gpt-4-0125-preview": 131072,
            "llama-2-7b-chat-hf": 4096,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "o3-mini": 200000,
            "deepseek-chat": 65536,
            "deepseek-reasoner": 65536,
            "qwen2.5-32b-instruct": 32768,
            "s1.1-32b": 32768,
            "deepseek-r1-distill-qwen-32b": 32768,
            "d1-32b": 32768,
            "m1-32b": 32768,
        }
        return send_token_limit_dict[model] if model in send_token_limit_dict else 8192

    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        if self.is_azure:
            openai_client = AzureOpenAI(
                api_key=self.client_args["api_key"],
                azure_endpoint=self.client_args["base_url"],
                api_version="2024-02-15-preview",
            )
        else:
            openai_client = OpenAI(
                api_key=self.client_args["api_key"],
                base_url=self.client_args["base_url"],
            )
        try:
            if functions != []:
                response = openai_client.chat.completions.create(
                    messages=messages,
                    functions=functions,
                    **self.args.dict(),
                )

                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    self.collect_metrics(response)

                    return LLMResult(
                        content=response.choices[0].message.get("content", ""),
                        function_name=response.choices[0].message.function_call.name,
                        function_arguments=ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        ),
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

            else:
                response = openai_client.chat.completions.create(
                    messages=messages,
                    **self.args.dict(),
                )
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                self.collect_metrics(response)
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise


    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        if self.args.model in ('deepseek-chat', 'deepseek-reasoner'):
            self.client_args["api_key"] = TOGETHER_API_KEY
            self.client_args["base_url"] = TOGETHER_BASE_URL
        elif self.args.model in LOCAL_LLMS:
            self.client_args["base_url"] = LOCAL_LLMS_MAPPING[self.args.model.split('/')[-1]]["base_url"]

        if self.is_azure:
            async_openai_client = AsyncAzureOpenAI(
                api_key=self.client_args["api_key"],
                azure_endpoint=self.client_args["base_url"],
                api_version="2024-02-15-preview",
            )
        else:
            async_openai_client = AsyncOpenAI(
                api_key=self.client_args["api_key"],
                base_url=self.client_args["base_url"],
                timeout=3000,
            )
        try:
            if functions != []:
                self.invert()
                response = await async_openai_client.chat.completions.create(
                    messages=messages,
                    functions=functions,
                    **self.args.dict(),
                )
                await async_openai_client.close()
                self.revert()
                response.choices[0].message.content = extract_reasoner(response.choices[0].message.content)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                if response.choices[0].message.function_call is not None:
                    function_name = response.choices[0].message.function_call.name
                    valid_function = False
                    if function_name.startswith("function."):
                        function_name = function_name.replace("function.", "")
                    elif function_name.startswith("functions."):
                        function_name = function_name.replace("functions.", "")
                    for function in functions:
                        if function["name"] == function_name:
                            valid_function = True
                            break
                    if not valid_function:
                        logger.warn(
                            f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                        )
                        raise ValueError(
                            f"The returned function name {function_name} is not in the list of valid functions."
                        )
                    try:
                        arguments = ast.literal_eval(
                            response.choices[0].message.function_call.arguments
                        )
                    except:
                        try:
                            arguments = ast.literal_eval(
                                JsonRepair(
                                    response.choices[0].message.function_call.arguments
                                ).repair()
                            )
                        except:
                            logger.warn(
                                "The returned argument in function call is not valid json. Retrying..."
                            )
                            raise ValueError(
                                "The returned argument in function call is not valid json."
                            )
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        function_name=function_name,
                        function_arguments=arguments,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

                else:
                    self.collect_metrics(response)
                    logger.log_prompt(
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    )
                    return LLMResult(
                        content=response.choices[0].message.content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )

            else:
                self.invert()
                response = await async_openai_client.chat.completions.create(
                    messages=messages,
                    **self.args.dict(),
                )
                await async_openai_client.close()
                self.revert()
                response.choices[0].message.content = extract_reasoner(response.choices[0].message.content)
                self.collect_metrics(response)
                logger.log_prompt(
                    [
                        {
                            "role": "assistant",
                            "content": response.choices[0].message.content,
                        }
                    ]
                )
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except Exception as e:
            await async_openai_client.close()
            self.revert()
            logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            raise

    def agenerate_response_local(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = []
    ) -> LLMResult:
        from vllm import SamplingParams, LLM
        # Construct prompt using system message and history
        messages = self.construct_messages(prepend_prompt, history, append_prompt)    
        self.invert()
        model_name = str(self.args.model)
        if model_name not in OpenAIChat.model_cache:
            OpenAIChat.model_cache[model_name] = LLM(
                model_name,
                tensor_parallel_size=4,
                enforce_eager=True,
                trust_remote_code=True,
                gpu_memory_utilization=0.95,
            )
        model = OpenAIChat.model_cache[model_name]
        tok = AutoTokenizer.from_pretrained(model_name)
        self.revert()
        
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        send_tokens = len(tok(prompt)["input_ids"])
        if self.args.model == 'd1-32b' or self.args.model == 'm1-32b':
            prompt += "<think>"
            output = "<think>"
        else:
            prompt += ""
            output = ""
        # Budget forcing logic
        stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
        sampling_params = SamplingParams(
            max_tokens=self.args.max_tokens,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            frequency_penalty=self.args.frequency_penalty,
            presence_penalty=self.args.presence_penalty,
        )
        
        try:
            o = model.generate(prompt, sampling_params)[0]
            output += o.outputs[0].text
            if count_string_tokens(o.outputs[0].text, self.args.model) >= self.args.max_tokens - 800:
                if self.args.model == 'd1-32b' or self.args.model == 'm1-32b':
                    prompt += o.outputs[0].text + "</think>"
                    output += "</think>"
                else:
                    prompt += o.outputs[0].text + "<Final Answer>:"
                    output += "<Final Answer>:"
                stop_token_ids = tok("<|im_end|>")["input_ids"]
                sampling_params = SamplingParams(
                    max_tokens=min(32768 - self.args.max_tokens, int(self.args.max_tokens * 0.5)),
                    min_tokens=0,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                )
                o = model.generate(prompt, sampling_params)[0]
                retry_cnt = 0
                while o.outputs[0].text == "" and (self.args.model == "d1-32b" or self.args.model == "m1-32b") and retry_cnt < 3:
                    logger.info("the initial content after thinking or final answer is empty, retrying...")
                    logger.info(f"input tokens of this request: {len(tok(prompt)['input_ids'])}")
                    invert_args(self.args, {})
                    if retry_cnt == 0:
                        prompt += "<Final Answer>:"
                        output += "<Final Answer>:"
                    o = model.generate(prompt, sampling_params)[0]
                    retry_cnt += 1
                output += o.outputs[0].text
                logger.info(f"budget forcing output below ==============================================")
                logger.info(f"thinking exceed {self.args.max_tokens}, content after thinking or final answer:\n{o.outputs[0].text}\n\n")
            output = extract_reasoner(output)

            return LLMResult(
                content=output,
                send_tokens=send_tokens,
                recv_tokens=len(tok(output)["input_ids"]),
                total_tokens=send_tokens + len(tok(output)["input_ids"])
            )
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            self.revert()
            raise


    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        if self.args.model in LOCAL_LLMS and len(history) > 0:
            if prepend_prompt != "":
                messages.append({"role": "system", "content": prepend_prompt})
            history_content = "Here is the chat history: <history>\n\n\n"
            for item in history:
                history_content += item["content"]
                history_content += "\n\n"
            messages.append({"role": "user", "content": history_content + '</history>\n\n\n' + append_prompt})
        else:
            if prepend_prompt != "":
                messages.append({"role": "system", "content": prepend_prompt})
            if len(history) > 0:
                messages += history
            if append_prompt != "":
                messages.append({"role": "user", "content": append_prompt})

        return messages

    def collect_metrics(self, response):
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

    def get_spend(self) -> int:
        input_cost_map = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-3.5-turbo-1106": 0.0005,
            "gpt-3.5-turbo-0125": 0.0005,
            "gpt-4": 0.03,
            "gpt-4-0613": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-4-1106-preview": 0.01,
            "gpt-4-0125-preview": 0.01,
            "llama-2-7b-chat-hf": 0.0,
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "o3-mini": 0.0011,
            "deepseek-chat": 0.00125,
            "deepseek-reasoner": 0.003,
            "qwen2.5-32b-instruct": 0,
            "s1.1-32b": 0,
            "deepseek-r1-distill-qwen-32b": 0,
            "d1-32b": 0,
            "m1-32b": 0,
        }

        output_cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-3.5-turbo-1106": 0.0015,
            "gpt-3.5-turbo-0125": 0.0015,
            "gpt-4": 0.06,
            "gpt-4-0613": 0.06,
            "gpt-4-32k": 0.12,
            "gpt-4-1106-preview": 0.03,
            "gpt-4-0125-preview": 0.03,
            "llama-2-7b-chat-hf": 0.0,
            "gpt-4o": 0.010,
            "gpt-4o-mini": 0.0006,
            "o3-mini": 0.0044,
            "deepseek-chat": 0.00125,
            "deepseek-reasoner": 0.007,
            "qwen2.5-32b-instruct": 0,
            "s1.1-32b": 0,
            "deepseek-r1-distill-qwen-32b": 0,
            "d1-32b": 0,
            "m1-32b": 0,
        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            raise ValueError(f"Model type {model} not supported")

        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def get_embedding(text: str, attempts=3) -> np.array:
    if AZURE_API_KEY and AZURE_API_BASE:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
    elif OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        ).model_dump_json(indent=2)
        return tuple(embedding)
    except Exception as e:
        attempt += 1
        logger.error(f"Error {e} when requesting openai models. Retrying")
        raise

