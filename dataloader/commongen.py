from .dataloader import DataLoader
from . import dataloader_registry
import json


@dataloader_registry.register("tasksolving/commongen/gpt-4")
@dataloader_registry.register("tasksolving/commongen/gpt-3.5")
@dataloader_registry.register("tasksolving/commongen/llama-2-7b-chat-hf")
@dataloader_registry.register("tasksolving/commongen/gpt-4o")
@dataloader_registry.register("tasksolving/commongen/gpt-4o-mini")
@dataloader_registry.register("tasksolving/commongen/o3-mini")
@dataloader_registry.register("tasksolving/commongen/deepseek-chat")
@dataloader_registry.register("tasksolving/commongen/deepseek-reasoner")
@dataloader_registry.register("tasksolving/commongen/qwen2.5-32b-instruct")
@dataloader_registry.register("tasksolving/commongen/s1.1-32b")
@dataloader_registry.register("tasksolving/commongen/deepseek-r1-distill-qwen-32b")
@dataloader_registry.register("tasksolving/commongen/d1-32b")
@dataloader_registry.register("tasksolving/commongen/m1-32b")
class CommongenLoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["concepts"],
                        "answer": None,
                    }
                )
