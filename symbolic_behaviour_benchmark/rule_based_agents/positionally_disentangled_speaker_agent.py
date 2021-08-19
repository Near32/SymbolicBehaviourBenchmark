from typing import List, Optional, Any, Dict
import numpy as np
import random
import copy

class PositionallyDisentangledSpeakerAgent(object):
    def __init__(
        self,
        action_space_dim:object, 
        vocab_size:int,
        max_sentence_length:int,
        nbr_communication_rounds:int,
        nbr_latents:int,
        ):
        self.action_space_dim = action_space_dim
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.nbr_communication_rounds = nbr_communication_rounds
        self.nbr_latents = nbr_latents
        
        self.reset()

    def reset(self):
        self.round_idx = 0
    
    def _reg_comm_chan(self, action_dict:Dict[str,np.ndarray])->Dict[str,np.ndarray]:
        comm_chan_reg_action_dict = copy.deepcopy(action_dict)
        comm_chan_reg_action_dict['communication_channel'] = action_dict['communication_channel'][0,self.round_idx:self.round_idx+self.max_sentence_length]
        return comm_chan_reg_action_dict
    
    def _utter(
        self,
        state:np.ndarray,
        infos:Dict[str,np.ndarray],
        )->Dict[str,np.ndarray]:
        
        action_dict = {
            "communication_channel":np.zeros((1,max(self.nbr_latents,self.max_sentence_length))),
            "decision":np.zeros((1,1)),
        }
        
        target_stimulus = infos["speaker_exp_latents"][0,0]

        for sid in range(self.nbr_latents):
            # watch out for eos_token whose index is 0...
            assert target_stimulus[sid]+1 < self.vocab_size
            action_dict["communication_channel"][0,sid] = target_stimulus[sid]+1

        return action_dict

    def next_action(
        self,
        state:np.ndarray,
        infos:Dict[str,np.ndarray],
        )->Dict[str,np.ndarray]:
        
        self.round_idx = infos['round_idx']

        if self.round_idx==0:
            """
            Compute communication_channel content:
            """
            self.action_dict = self._utter(state=state, infos=infos)
        
        if self.round_idx!=self.nbr_communication_rounds:
            comm_chan_reg_action_dict = self._reg_comm_chan(self.action_dict)
        else:
            comm_chan_reg_action_dict = {
                "communication_channel":np.zeros((1, self.max_sentence_length)),
                "decision":np.zeros((1,1)),
            }
        

        return comm_chan_reg_action_dict


from ..utils.agent_wrappers import RuleBasedAgentWrapper

def build_WrappedPositionallyDisentangledSpeakerAgent(
        player_idx:int, 
        action_space_dim:object, 
        vocab_size:int,
        max_sentence_length:int,
        nbr_communication_rounds:int,
        nbr_latents:int,
        ):
	agent = PositionallyDisentangledSpeakerAgent(
            action_space_dim=action_space_dim, 
            vocab_size=vocab_size,
            max_sentence_length=max_sentence_length,
            nbr_communication_rounds=nbr_communication_rounds,
            nbr_latents=nbr_latents,
        )
	wrapped_agent = RuleBasedAgentWrapper(
		ruleBasedAgent=agent, 
		player_idx=player_idx, 
		nbr_actors = 1
	)
	return wrapped_agent
