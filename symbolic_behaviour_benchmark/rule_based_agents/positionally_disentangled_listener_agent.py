from typing import List, Optional, Any, Dict
import numpy as np
import random
import copy


class PositionallyDisentangledListenerAgent(object):
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
    
    def _reason(
        self,
        state:np.ndarray,
        infos:Dict[str,np.ndarray],
        action_dict:Optional[Dict[str,np.ndarray]]=None,
        )->Dict[str,np.ndarray]:
        
        if action_dict is None:
            action_dict = {
                "communication_channel":np.zeros((1,self.max_sentence_length)),
                "decision":np.zeros((1,1)),
            }
        
        target_stimulus = infos["speaker_exp_latents"][0:1,0:1]
        target_utterance_ohe = infos["communication_channel"]
        target_utterance_widx = np.reshape(target_utterance_ohe, (self.max_sentence_length,-1))
        target_utterance_widx = (np.arange(self.vocab_size+1)*target_utterance_widx).max(axis=-1)

        stimuli = infos["listener_exp_latents"][0]
        
        target_utterance_round_pos_start = self.round_idx-1
        target_utterance_round_pos_end = self.round_idx
        if self.round_idx == 1 and self.nbr_communication_rounds==1:
            target_utterance_round_pos_start = 0
            target_utterance_round_pos_end = self.max_sentence_length

        round_target_utterance = \
            target_utterance_widx[target_utterance_round_pos_start:target_utterance_round_pos_end]
        
        round_stimuli = stimuli[..., target_utterance_round_pos_start:target_utterance_round_pos_end]
        
        """
        Taking care of EoS offset index:
        """
        round_target_utterance_widx = target_utterance_widx-1
        rtu_widx = np.stack([round_target_utterance_widx]*round_stimuli.shape[0], axis=0)
        # nbr_stimulus x stimulus_dim_we_care_about 
        round_stimuli_scores = (round_stimuli==rtu_widx).astype(float).sum(axis=-1)
        round_decision = round_stimuli_scores.argmax(axis=0).astype(float)
        action_dict["decision"][0,0] = round_decision
        
        return action_dict

    def next_action(
        self,
        state:np.ndarray,
        infos:Dict[str,np.ndarray],
        )->Dict[str,np.ndarray]:
        
        self.round_idx = infos['round_idx']
        
        self.action_dict = {
            "communication_channel":np.zeros((1,self.max_sentence_length)),
            "decision":np.zeros((1,1)),
        }
        
        if self.round_idx!=0:
            self.action_dict = self._reason(
                state=state, 
                infos=infos,
                action_dict=self.action_dict,
            )

            self.per_round_decision.append(self.action_dict['decision'])
        else:
            self.per_round_decision = []

        if self.round_idx==self.nbr_communication_rounds:
            """
            It is time to make the final decision:
            """
            final_decision = random.choice(self.per_round_decision)
            self.action_dict['decision'] = final_decision

        return self.action_dict


from ..utils.agent_wrappers import RuleBasedAgentWrapper

def build_WrappedPositionallyDisentangledListenerAgent(
        player_idx:int, 
        action_space_dim:object, 
        vocab_size:int,
        max_sentence_length:int,
        nbr_communication_rounds:int,
        nbr_latents:int,
        ):
	agent = PositionallyDisentangledListenerAgent(
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
