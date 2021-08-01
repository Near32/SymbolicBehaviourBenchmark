from .envs import *
from .utils import * 

import gym
from gym.envs.registration import register

env_dict = gym.envs.registration.registry.env_specs.copy()

for env in env_dict:
    if 'SymbolicBehaviourBenchmark' in env:
        del gym.envs.registration.registry.env_specs[env]

register(
    id='SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0',
    entry_point='symbolic_behaviour_benchmark.envs:generate_receptive_constructive_test_env'
)