from .dataloader import DataLoader
from . import dataloader_registry
import json
import re


@dataloader_registry.register("tasksolving/gpqa_diamond/gpt-4")
@dataloader_registry.register("tasksolving/gpqa_diamond/gpt-3.5")
@dataloader_registry.register("tasksolving/gpqa_diamond/gpt-4o")
@dataloader_registry.register("tasksolving/gpqa_diamond/gpt-4o-mini")
@dataloader_registry.register("tasksolving/gpqa_diamond/o3-mini")
@dataloader_registry.register("tasksolving/gpqa_diamond/deepseek-chat")
@dataloader_registry.register("tasksolving/gpqa_diamond/deepseek-reasoner")
@dataloader_registry.register("tasksolving/gpqa_diamond/qwen2.5-32b-instruct")
@dataloader_registry.register("tasksolving/gpqa_diamond/s1.1-32b")
@dataloader_registry.register("tasksolving/gpqa_diamond/deepseek-r1-distill-qwen-32b")
@dataloader_registry.register("tasksolving/gpqa_diamond/d1-32b")
@dataloader_registry.register("tasksolving/gpqa_diamond/m1-32b")
class GPQADiamondLoader(DataLoader):
    def __init__(self, path: str):
        self.answer_pat = re.compile(r"#### (-?\d+)")
        super().__init__(path)

    def load(self):
        with open(self.path) as f:
            for line in f:
                line = json.loads(line)
                self.examples.append(
                    {
                        "input": line["Question_w_answers"],
                        "answer": f"B): {line['Correct Answer']}",
                    }
                )
