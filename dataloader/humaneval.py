from .dataloader import DataLoader
from . import dataloader_registry
import json


@dataloader_registry.register("tasksolving/humaneval/gpt-4")
@dataloader_registry.register("tasksolving/humaneval/gpt-3.5")
@dataloader_registry.register("tasksolving/humaneval/gpt-4o")
@dataloader_registry.register("tasksolving/humaneval/gpt-4o-mini")
@dataloader_registry.register("tasksolving/humaneval/o3-mini")
@dataloader_registry.register("tasksolving/humaneval/deepseek-chat")
@dataloader_registry.register("tasksolving/humaneval/deepseek-reasoner")
@dataloader_registry.register("tasksolving/humaneval/qwen2.5-32b-instruct")
@dataloader_registry.register("tasksolving/humaneval/s1.1-32b")
@dataloader_registry.register("tasksolving/humaneval/deepseek-r1-distill-qwen-32b")
@dataloader_registry.register("tasksolving/humaneval/d1-32b")
@dataloader_registry.register("tasksolving/humaneval/m1-32b")
class HumanevalLoader(DataLoader):
    def __init__(self, path: str):
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["prompt"],
                        "answer": line["test"],
                    }
                )
