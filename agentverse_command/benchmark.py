import logging
import os
import json
import shutil
from colorama import Fore, init
import yaml
init(autoreset=True)

from agentverse.tasksolving import TaskSolving
from agentverse.logging import get_logger
from argparse import ArgumentParser
import asyncio
from dataloader import dataloader_registry

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = ArgumentParser()

parser.add_argument("--task", type=str, default="tasksolving/gpqa_diamond/gpt-4o-mini")
parser.add_argument("--tasks_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "agentverse", "tasks"))
parser.add_argument("--dataset_path", type=str, default="data/gpqa_diamond/test.jsonl")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--has_tools", action="store_true")
parser.add_argument("--tool_tmp_path", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--single_agent", action="store_true")
parser.add_argument("--collect", action="store_false", dest="collect", help="Disable overwrite mode")
args = parser.parse_args()

logger = get_logger()
logger.set_level(logging.DEBUG if args.debug else logging.INFO)
logger.info(Fore.YELLOW + json.dumps(args.__dict__, indent=4))


def get_dataloader(task, dataset_path):
    return dataloader_registry.build(task, path=dataset_path)


def cli_main():
    dataloader = get_dataloader(args.task, args.dataset_path)
    if args.output_path is None:
        os.makedirs(f"./results/{args.task}", exist_ok=True)
        args.output_path = f"./results/{args.task}"
    else:
        args.output_path = f"./results/{args.task}/{args.output_path}"
        os.makedirs(args.output_path, exist_ok=True)

    shutil.copyfile(
        f"{args.tasks_dir}/{args.task}/config.yaml",
        f"{args.output_path}/config.yaml",
    )

    skip_cnt = 0
    if not args.overwrite and os.path.exists(f"{args.output_path}/results.jsonl"):
        with open(f"{args.output_path}/results.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    skip_cnt += 1
    f = open(f"{args.output_path}/results.jsonl", "w" if args.overwrite else "a")
    if args.debug:
        skip_cnt = 0
    for i, example in enumerate(dataloader):
        if i < skip_cnt:
            continue
        logger.info(f"Index: {i}, Input: {example['input']}\nAnswer: {example['answer']}")
        if args.has_tools:
            assert args.tool_tmp_path is not None
            with open(args.tool_tmp_path, "w") as tool_f:
                tool_f.write(json.dumps(example["tools"]))
        agentverse = TaskSolving.from_task(args.task, args.output_path)
        agentverse.environment.set_task_description(example["input"])
        agentverse.environment.single_agent = args.single_agent
        plan, result, logs = agentverse.run()
        f.write(
            json.dumps(
                {
                    "input": example["input"],
                    "response": plan,
                    "label": example["answer"],
                    "spend": agentverse.environment.get_spend(),
                    "logs": logs,
                }
            )
            + "\n"
        )
        f.flush()
    f.close()


if __name__ == "__main__":
    cli_main()
