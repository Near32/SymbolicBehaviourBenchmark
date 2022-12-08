import copy
import numpy as np
import gym


class DiscreteCombinedActionWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment's action space is a Dict that contains 
    the keys "communication_channel" and "decision".
    Firstly, it combines both spaces into a Discrete action space, and adds a No-op action
    at the end for the player who cannot play the current turn.
    Secondly, it augments the infos list of dictionnary with entries 
    "legal_actions" and "action_mask", for each player's info. 
    Args:
        - env (gym.Env): Env to wrap.
    """
    def __init__(self, env):
        super(DiscreteCombinedActionWrapper, self).__init__(env)
        self.wrapped_action_space = env.action_space 
        
        self.vocab_size = self.wrapped_action_space.spaces["communication_channel"].vocab_size
        self.max_sentence_length = self.wrapped_action_space.spaces["communication_channel"].max_sentence_length

        self.nb_decisions = self.wrapped_action_space.spaces["decision"].n 
        
        self._build_sentenceId2sentence()
        
        # Action Space:
        self.nb_possible_actions = self.nb_decisions*self.nb_possible_sentences
        # Adding no-op action:
        self.action_space = gym.spaces.Discrete(self.nb_possible_actions+1)

        self.observation_space = env.observation_space

    def _build_sentenceId2sentence(self):
        self.nb_possible_sentences = 1 # empty string...
        for pos in range(self.max_sentence_length):
            # account for each string of length pos (before EoS)
            self.nb_possible_sentences += (self.vocab_size)**(pos+1)
        
        sentenceId2sentence = np.zeros( (self.nb_possible_sentences, self.max_sentence_length))
        idx = 1
        local_token_pointer = 0
        global_token_pointer = 0
        while idx != self.nb_possible_sentences:
            sentenceId2sentence[idx] = sentenceId2sentence[idx-1]
            sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)

            while sentenceId2sentence[idx][local_token_pointer] == 0:
                # remove the possibility of an empty symbol on the left of actual tokens:
                sentenceId2sentence[idx][local_token_pointer] += 1
                local_token_pointer += 1
                sentenceId2sentence[idx][local_token_pointer] = (sentenceId2sentence[idx][local_token_pointer]+1)%(self.vocab_size+1)
            idx += 1
            local_token_pointer = 0    
        
        self.sentenceId2sentence = sentenceId2sentence

        self.sentence2sentenceId = {}
        for sid in range(self.nb_possible_sentences):
            self.sentence2sentenceId[ self.sentenceId2sentence[sid].tostring() ] = sid        
        
    def _make_infos(self, observations, infos):
        self.infos = []

        # Adapt info's legal_actions:
        for player_idx in range(self.nbr_agent):
            # Only No-op:
            legal_moves= [self.action_space.n-1]
            # unless speaker role 
            # or second round of the referential game, when the listener makes a prediction:
            # this round_idx has NOT been updated yet,
            # thus it really describe the idx of the current round: 
            if player_idx==0\
            or infos[player_idx]['round_idx']==infos[player_idx]["nbr_communication_rounds"]: 
                # Everything actually available, except No-op:
                legal_moves = list(range(self.action_space.n-1))
            
            action_mask=np.zeros((1,self.action_space.n))
            np.put(action_mask, ind=legal_moves, v=1)
            
            info = copy.deepcopy(infos[player_idx])
            info['action_mask'] = action_mask
            info['legal_actions'] = action_mask
            self.infos.append(info)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        
        self.nbr_agent = len(infos)
        
        self._make_infos(observations, infos)

        return observations, copy.deepcopy(self.infos) 

    def _decode_action(self, action):
        action_dicts = []
        for pidx in range(self.nbr_agent):
            pidx_a = action[pidx]
            if isinstance(pidx_a, np.ndarray):  pidx_a = pidx_a.item()
            
            if not self.action_space.contains(pidx_a):
                raise ValueError('action {} is invalid for {}'.format(pidx_a, self.action_space))

            if pidx_a<(self.action_space.n-1):
                original_action_decision_id = pidx_a // self.nb_possible_sentences
                
                original_action_sentence_id = (pidx_a % self.nb_possible_sentences)
                original_action_sentence = self.sentenceId2sentence[original_action_sentence_id:original_action_sentence_id+1] 
                # batch=1 x max_sentence_length
            else:
                # No-op action ==> decision=0 + EoS message
                original_action_decision_id = 0
                original_action_sentence_id = 0
                original_action_sentence = self.sentenceId2sentence[original_action_sentence_id:original_action_sentence_id+1]
            
            ad = {
                'decision':original_action_decision_id,
                'communication_channel':original_action_sentence
            }

            action_dicts.append(ad)
        
        return action_dicts

    def _encode_action(self, action_dict):
        original_action_decision_id = action_dict['decision']
        original_action_sentence = action_dict['communication_channel']
        
        original_action_sentence_id = self.sentence2sentenceId[ original_action_sentence.tostring() ]

        # Is it No-op?
        if original_action_sentence[0].item()==0 and original_action_decision_id==0:
            encoded_action = self.action_space.n-1
        else:
            encoded_action = original_action_decision_id*self.nb_possible_sentences+original_action_sentence_id

        return encoded_action
    
    def step(self, action):
        original_action = self._decode_action(action)

        next_observations, reward, done, next_infos = self.env.step(original_action)

        self.nbr_agent = len(next_infos)
        
        self._make_infos(next_observations, next_infos)

        return next_observations, reward, done, copy.deepcopy(self.infos)


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
            for token_idx in observations[player_idx]["communication_channel"][0]:
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

class StimulusObservationWrapper(gym.Wrapper):
    """
    Assumes the :arg env: environment to have a Dict observation space,
    that contains the key 'stimulus'.
    This wrapper makes the observation space consisting of solely the 'stimulus' entry,
    while the other entries are put in the infos dictionnary.
    Args:
        env (gym.Env): Env to wrap.
    """

    def __init__(self, env):
        super(StimulusObservationWrapper, self).__init__(env)
        
        self.observation_space = env.observation_space.spaces["stimulus"]

        self.action_space = env.action_space 

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        nbr_agent = len(infos)
        
        new_observations = [obs["stimulus"] for obs in observations]

        for agent_idx in range(nbr_agent):
            oobs = observations[agent_idx]

            for k,v in oobs.items():
                if k=="stimulus":  continue
                infos[agent_idx][k] = v

        return new_observations, infos 
    
    def step(self, action):
        next_observations, reward, done, next_infos = self.env.step(action)        
        nbr_agent = len(next_infos)
        
        new_next_observations = [obs["stimulus"] for obs in next_observations]

        for agent_idx in range(nbr_agent):
            oobs = next_observations[agent_idx]

            for k,v in oobs.items():
                if k=="stimulus":  continue
                next_infos[agent_idx][k] = v
        
        return new_next_observations, reward, done, next_infos

    def render(self, mode='human', **kwargs):
        env = self.unwrapped
        return env.render(
            mode=mode,
            **kwargs,
        )
        
def s2b_wrap(env, combined_actions=False, dict_obs_space=False, multi_binary_comm=False):
    if combined_actions:
        env = DiscreteCombinedActionWrapper(env)
    if multi_binary_comm \
    and any([("communication_channel" in k) for k in env.unwrapped.observation_space]):
        env = MultiBinaryCommunicationChannelWrapper(env)
    if not dict_obs_space:
        env = StimulusObservationWrapper(env)
    return env
