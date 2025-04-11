from .dataloader import DataLoader
from . import dataloader_registry
import json


@dataloader_registry.register("tasksolving/mbpp_sanitized/gpt-4")
@dataloader_registry.register("tasksolving/mbpp_sanitized/gpt-3.5")
@dataloader_registry.register("tasksolving/mbpp_sanitized/gpt-4o")
@dataloader_registry.register("tasksolving/mbpp_sanitized/gpt-4o-mini")
@dataloader_registry.register("tasksolving/mbpp_sanitized/o3-mini")
@dataloader_registry.register("tasksolving/mbpp_sanitized/deepseek-chat")
@dataloader_registry.register("tasksolving/mbpp_sanitized/deepseek-reasoner")
@dataloader_registry.register("tasksolving/mbpp_sanitized/qwen2.5-32b-instruct")
@dataloader_registry.register("tasksolving/mbpp_sanitized/s1.1-32b")
@dataloader_registry.register("tasksolving/mbpp_sanitized/deepseek-r1-distill-qwen-32b")
@dataloader_registry.register("tasksolving/mbpp_sanitized/d1-32b")
@dataloader_registry.register("tasksolving/mbpp_sanitized/m1-32b")
class MBPPSanitizedLoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["prompt_w_code_header"],
                        "answer": line["test_list"],
                    }
                )
