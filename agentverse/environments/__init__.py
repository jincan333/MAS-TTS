from typing import Dict
from agentverse.registry import Registry


env_registry = Registry(name="EnvironmentRegistry")


from .base import BaseEnvironment, BaseRule

# from .basic import PipelineEnvironment

from .tasksolving_env.basic import BasicEnvironment
