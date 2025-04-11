from agentverse.registry import Registry

dataloader_registry = Registry(name="dataloader")

from .humaneval import HumanevalLoader
from .commongen import CommongenLoader
from .gpqa_diamond import GPQADiamondLoader
from .math500 import MATH500Loader
from .aime2024 import AIME2024Loader
from .mbpp_sanitized import MBPPSanitizedLoader
