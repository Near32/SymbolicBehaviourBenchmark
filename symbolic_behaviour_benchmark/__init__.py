from .envs import *
from .utils import * 
from .rule_based_agents import * 

import gym
from gym.envs.registration import register

for env_k in gym.envs.registration.registry.env_specs.keys():
    if 'SymbolicBehaviourBenchmark' in env_k:
        del gym.envs.registration.registry.env_specs[env_k]

register(
    id='SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0',
    entry_point='symbolic_behaviour_benchmark.envs:generate_receptive_constructive_test_env'
)

register(
    id='SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-2Shots-v0',
    entry_point='symbolic_behaviour_benchmark.envs:generate_receptive_constructive_test_env_2shots'
)


register(
    id='SymbolicBehaviourBenchmark-RecallTestEnv-v0',
    entry_point='symbolic_behaviour_benchmark.envs:generate_recall_test_env'
)

register(
    id='SymbolicBehaviourBenchmark-RecallTestEnv-2Shots-v0',
    entry_point='symbolic_behaviour_benchmark.envs:generate_recall_test_env_2shots'
)

