from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/math500/gpt-4")
@dataloader_registry.register("tasksolving/math500/gpt-3.5")
@dataloader_registry.register("tasksolving/math500/gpt-4o")
@dataloader_registry.register("tasksolving/math500/gpt-4o-mini")
@dataloader_registry.register("tasksolving/math500/o3-mini")
@dataloader_registry.register("tasksolving/math500/deepseek-chat")
@dataloader_registry.register("tasksolving/math500/deepseek-reasoner")
@dataloader_registry.register("tasksolving/math500/qwen2.5-32b-instruct")
@dataloader_registry.register("tasksolving/math500/s1.1-32b")
@dataloader_registry.register("tasksolving/math500/deepseek-r1-distill-qwen-32b")
@dataloader_registry.register("tasksolving/math500/d1-32b") 
@dataloader_registry.register("tasksolving/math500/m1-32b")
class MATH500Loader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["problem"],
                        "answer": line["answer"],
                    }
                )
