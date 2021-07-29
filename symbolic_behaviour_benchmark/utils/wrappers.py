import copy
import numpy as np
import gym


class MultiBinaryCommunicationChannelWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment to have a Dict observation space,
    that contains the key "communication_channel", whose value is a MultiDiscrete.
    It transforms the MultiDiscrete observation in to a MultiBinary that is 
    the concatenation of each of the one-hot-encoded Discrete values.

    The communication channel allow for vocabulary_size ungrounded symbols
    and one grounded symbol that acts as EoS, whose index is 0.

    Args:
        env (gym.Env): Env to wrap. 
    """
    def __init__(self, env):
        super(MultiBinaryCommunicationChannelWrapper, self).__init__(env)

        self.observation_space = copy.deepcopy(env.observation_space)
        self.vocabulary_size = self.observation_space.spaces["communication_channel"].vocab_size
        self.max_sentence_length = self.observation_space.spaces["communication_channel"].max_sentence_length
        self.communication_channel_observation_space_size = self.max_sentence_length*(self.vocabulary_size+1)
        self.observation_space.spaces["communication_channel"] = gym.spaces.Discrete(self.communication_channel_observation_space_size)

        self.action_space = self.env.action_space 

    def _make_obs_infos(self, observations, infos):
        for player_idx in range(len(observations)):        
            token_start = 0
            new_communication_channel = np.zeros((1, self.communication_channel_observation_space_size))
            for token_idx in observations[player_idx]["communication_channel"]:
                new_communication_channel[0, int(token_start+token_idx)] = 1
                token_start += self.vocabulary_size+1
            observations[player_idx]["communication_channel"] = new_communication_channel
        return observations, infos

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        observations, infos = self._make_obs_infos(
            observations=observations,
            infos=infos,
        )
        return observations, infos

    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)
        next_observations, next_infos = self._make_obs_infos(
            observations=next_observations,
            infos=next_infos,
        )
        return next_observations, reward, done, next_infos