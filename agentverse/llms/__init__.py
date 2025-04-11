from agentverse.registry import Registry
import os

llm_registry = Registry(name="LLMRegistry")
LOCAL_LLMS = [
    "llama-2-7b-chat-hf",
    "llama-2-13b-chat-hf",
    "llama-2-70b-chat-hf",
    "vicuna-7b-v1.5",
    "vicuna-13b-v1.5",
    "qwen2.5-32b-instruct",
    "s1.1-32b",
    "deepseek-r1-distill-qwen-32b",
    "d1-32b",
    "Qwen/Qwen2.5-32B-Instruct",
    "simplescaling/s1.1-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "m1-32b",
]
LOCAL_LLMS.append(os.environ.get("model_name_or_path"))

LOCAL_LLMS_MAPPING = {
    "llama-2-7b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "llama-2-13b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-13b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "llama-2-70b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-70b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "vicuna-7b-v1.5": {
        "hf_model_name": "lmsys/vicuna-7b-v1.5",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "vicuna-13b-v1.5": {
        "hf_model_name": "lmsys/vicuna-13b-v1.5",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "qwen2.5-32b-instruct": {
        "hf_model_name": "Qwen/Qwen2.5-32B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
    },
    "s1.1-32b": {
        "hf_model_name": "simplescaling/s1.1-32B",
        "base_url": "http://localhost:8001/v1",
        "api_key": "EMPTY",
    },
    "deepseek-r1-distill-qwen-32b": {
        "hf_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "base_url": "http://localhost:8002/v1",
        "api_key": "EMPTY",
    },
    "d1-32b": {
        "hf_model_name": os.environ.get("model_name_or_path"),
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY",
    },
    "m1-32b": {
        "hf_model_name": os.environ.get("model_name_or_path"),
        "base_url": "http://localhost:8003/v1",
        "api_key": "EMPTY",
    },
}


from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat
