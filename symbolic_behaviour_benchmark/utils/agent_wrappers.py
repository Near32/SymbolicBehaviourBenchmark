from typing import List, Optional
import copy
import numpy as np 

class RuleBasedAgentWrapper(object):
    def __init__(
        self, 
        ruleBasedAgent:object, 
        player_idx:int, 
        nbr_actors:int
        ):
        self.nbr_actors = nbr_actors
        self.action_space_dim = ruleBasedAgent.action_space_dim
        self.vocab_size = ruleBasedAgent.vocab_size
        self.max_sentence_length = ruleBasedAgent.max_sentence_length
        self.nbr_communication_rounds = ruleBasedAgent.nbr_communication_rounds
        self.nbr_latents = ruleBasedAgent.nbr_latents
        
        self.training = False
        self.player_idx = player_idx
        self.original_ruleBasedAgent = ruleBasedAgent
        self.ruleBasedAgents = []
        self.reset_actors()
        
        self.nb_possible_sentences = self.vocab_size**self.max_sentence_length
        
        self._build_sentenceId2sentence()
        
        self.nb_decisions = (self.action_space_dim-1)//self.nb_possible_sentences
        
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
        
    def _encode_action(self, action_dict, info_dict):
        original_action_decision_id = action_dict['decision']
        original_action_sentence = action_dict['communication_channel']
        
        original_action_sentence_id = self.sentence2sentenceId[ original_action_sentence.tostring() ]

        # Are there actions available apart from No-op?
        available_actions_ids_p1 = info_dict['legal_actions'][0]* np.arange(info_dict['legal_actions'].shape[-1]+1)[1:]
        available_actions_set = set(available_actions_ids_p1.astype(int))
        available_actions_set = available_actions_set.difference([0])
        available_actions = [a-1 for a in available_actions_set]

        if available_actions==[self.action_space_dim-1]:
            encoded_action = self.action_space_dim-1
        else:
            encoded_action = original_action_decision_id*self.nb_possible_sentences+original_action_sentence_id

        return encoded_action
    

    def clone(self, **kwargs):
        cloned_agent = copy.deepcopy(self)
        cloned_agent.reset_actors()
        return cloned_agent

    @property
    def handled_experiences(self):
        return 0

    @handled_experiences.setter
    def handled_experiences(self, val):
        pass

    def get_experience_count(self):
        return self.handled_experiences

    def get_update_count(self):
        return 0

    def get_nbr_actor(self) -> int:
        return self.nbr_actors

    def parameters(self):
        return []

    def set_nbr_actor(self, nbr_actors:int):
        self.nbr_actors = nbr_actors
        self.reset_actors()

    def get_rnn_states(self):
        return copy.deepcopy(self.ruleBasedAgents)

    def set_rnn_states(self, rnn_states):
        self.ruleBasedAgents = rnn_states

    def reset_actors(self, indices:List[int]=None):
        if indices is None: indices = list(range(self.nbr_actors))
        
        for idx in indices:
            if len(self.ruleBasedAgents) <= idx:
                self.ruleBasedAgents.append(copy.deepcopy(self.original_ruleBasedAgent))
                continue
            self.ruleBasedAgents[idx] = copy.deepcopy(self.original_ruleBasedAgent)
            self.ruleBasedAgents[idx].reset()
    
    def get_hidden_state(self):
        return [self.ruleBasedAgents[a].get_hidden_state() for a in range(self.nbr_actors)]

    def query_action(self, state, infos, as_logit=False):
        return self.take_action(state=state, infos=infos, as_logit=as_logit)
    
    def take_action(self, state, infos, as_logit=False):
        """
        Convert the :param state: and :param infos:
        into the input that the rule-based agent expects. 
        """

        actions = np.asarray([
            self.action_space_dim-1 for _ in range(self.nbr_actors)
            ]
        )
        
        for pidx in range(self.nbr_actors):
            next_action_dict = self.ruleBasedAgents[pidx].next_action(
                state=state[pidx], 
                infos=infos[pidx]
            )
            
            actions[pidx] = self._encode_action(action_dict=next_action_dict, info_dict=infos[pidx])
            
        return actions


