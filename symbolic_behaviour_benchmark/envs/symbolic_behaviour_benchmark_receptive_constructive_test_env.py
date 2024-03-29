from typing import Dict, List, Optional 

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

import gym
from gym import spaces
from gym.utils import seeding

import logging 
import copy

logger = logging.getLogger(__name__)

from symbolic_behaviour_benchmark.envs.communication_channel import CommunicationChannel 
from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import SymbolicContinuousStimulusDataset 

from symbolic_behaviour_benchmark.utils import DualLabeledDataset
from symbolic_behaviour_benchmark.utils import DictDatasetWrapper


class CommunicationChannelPermutation(object):
    def __init__(self, env, identity=False):
        self.env = env 
        self.identity = identity

        self.vocab_size = self.env.vocab_size
        self.max_sentence_length = self.env.max_sentence_length
        
        self.reset()

    def reset(self):
        # Communication Channel:
        shuffledarr = np.arange(start=1,stop=self.vocab_size+1)
        if not self.identity:
            np.random.shuffle(shuffledarr)
        
        # WARNING: idx 0 is the grounded EoS symbol:
        self.communication_channel_bijection_decoder = { idx+1: v.item() for idx, v in enumerate(shuffledarr)}
        self.communication_channel_bijection_decoder[0] = 0 
        self.communication_channel_bijection_encoder = dict(zip(self.communication_channel_bijection_decoder.values(), self.communication_channel_bijection_decoder.keys()))        

    def encode_obs(self, obs):
        """

        """
        self.previous_obs = copy.deepcopy(obs)
        self.new_obs = copy.deepcopy(obs)

        comm = copy.deepcopy(
            obs.get(
                "communication_channel", 
                np.zeros(shape=(1,self.max_sentence_length,), dtype=np.int64)
            )
        )
        for idx in range(self.max_sentence_length):
            comm[0,idx] = self.communication_channel_bijection_encoder[comm[0,idx].item()]
        self.new_obs["communication_channel"] = comm
        
        return copy.deepcopy(self.new_obs)

    def encode_info(self, info):
        """

        """
        self.previous_info = copy.deepcopy(info)
        self.new_info = copy.deepcopy(info)

        return copy.deepcopy(self.new_info)

    def decode_action(self, action):
        """
        :param Action: Dict that contains the keys:
            - "communication_channel": ... 
        """
        self.previous_action = copy.deepcopy(action)
        self.new_action = copy.deepcopy(action)

        # Communication Channel:
        comm = copy.deepcopy(
            action.get(
                "communication_channel", 
                np.zeros(shape=(1,self.max_sentence_length,), dtype=np.int64)
            )
        )
        for idx in range(self.max_sentence_length):
            comm[0,idx] = self.communication_channel_bijection_decoder[comm[0,idx].item()]
        self.new_action["communication_channel"] = comm 

        return copy.deepcopy(self.new_action)

    def encode_action(self, action):
        """
        :param Action: Dict that contains the keys:
            - "communication_channel": ... 
            corresponding to the action as seen by the agent.

        :return EncodedAction: Dict that contains the keys:
            - "communication_channel": ... 
            corresponding to the action as seen by the player.
        """
        previous_action = copy.deepcopy(action)
        new_action = copy.deepcopy(action)

        # Communication Channel:
        comm = copy.deepcopy(
            action.get(
                "communication_channel", 
                np.zeros(shape=(1,self.max_sentence_length, ), dtype=np.int64)
            )
        )
        for idx in range(self.max_sentence_length):
            comm[0,idx] = self.communication_channel_bijection_encoder[comm[0,idx].item()]
        new_action["communication_channel"] = comm 

        return copy.deepcopy(new_action)



class SymbolicBehaviourBenchmark_ReceptiveConstructiveTestEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        rg_config:Dict[str,str],
        datasets:Dict[str,DualLabeledDataset],
        seed=1337,
        allow_listener_query=False,
        provide_listener_feedback=False,
        use_communication_channel_permutations=True,
        nbr_shots=1,
        **kwargs,
    ):  
        super(SymbolicBehaviourBenchmark_ReceptiveConstructiveTestEnv, self).__init__()
        self.nbr_players = 2
        self.rg_config = rg_config
        self.datasets = datasets
        assert 'train' in self.datasets.keys()
        assert 'test' in self.datasets.keys()
        self.mode = 'train'
        
        self.nbr_communication_rounds = rg_config.get("nbr_communication_rounds",1)
        self.max_sentence_length = rg_config.get("max_sentence_length", 5)
        self.vocab_size = rg_config.get("vocab_size", 10)
        self.nbr_distractors = rg_config.get("nbr_distractors", 2)
        self.nbr_latents = rg_config.get("nbr_latents", None)
        self.allow_listener_query = allow_listener_query
        self.use_communication_channel_permutations = use_communication_channel_permutations
        self.nbr_shots = nbr_shots 
        self.listener_feedback = provide_listener_feedback
        self.feedback_provided = False

        # Actions consist of a dictionnary of two elements:
        # - decision that is discrete integer valued
        # - communication channel that consist of ungrounded tokens, represented as integer values.
        nbr_decisions = self.nbr_distractors+1
        if rg_config.get('descriptive', False): nbr_decisions += 1
        self.decision_space = spaces.Discrete(nbr_decisions)
        self.communication_channel_action_space = CommunicationChannel(
            max_sentence_length=self.max_sentence_length,
            vocab_size=self.vocab_size
        )
        self.action_space = spaces.Dict({
            'decision': self.decision_space,
            'communication_channel': self.communication_channel_action_space
        })
        
       
        # Observations are dictionaries containing:
        # -stimulus,
        # -other player id,
        # -previous referential game's reward,
        # -previous referential gamee's success boolean,
        # -a communication channel output (either from the speaker or listener agent).
        self.stimulus_observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=((self.nbr_distractors+1)*rg_config.get('nbr_stimulus', 1)*self.nbr_latents, ),
            dtype=np.float32
        )
        self.communication_channel_observation_space = copy.deepcopy(self.communication_channel_action_space)

        self.id_length = 10
        self.other_agent_id_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.id_length,),
            dtype='uint8'
        )
        self.role_id_observation_space = spaces.MultiBinary(n=2)
        # role id : toggle index 0==speaker / 1==listener
        
        self.mode_id_observation_space = spaces.MultiBinary(n=2)
        # mode id : toggle index between training/support==0 / test==1


        self.previous_game_reward_observation_space = spaces.Box(
            low=-10,
            high=10,
            shape=(1,),
            dtype=np.float32,
        )

        self.previous_game_result_observation_space = spaces.MultiBinary(n=2)
        # toggle index 0==failure / 1==success

        self.observation_space = spaces.Dict({
            'stimulus': self.stimulus_observation_space,
            'communication_channel': self.communication_channel_observation_space,
            'other_agent_id': self.other_agent_id_observation_space,
            'role_id': self.role_id_observation_space,
            'mode_id': self.mode_id_observation_space,
            'previous_game_reward': self.previous_game_reward_observation_space,
            'previous_game_result': self.previous_game_result_observation_space,
        })
        
        self.per_player_permutation = [
            CommunicationChannelPermutation(env=self, identity=not(self.use_communication_channel_permutations))
            for _ in range(self.nbr_players)
        ]

        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed 

    def _regularise_communication_channel(self, communication_channel_content):
        # Regularise the use of EoS symbol which is idx 0 of the vocabulary:
        make_eos = False
        # batch dim=1 x max_sentence_length...
        for idx, o in enumerate(communication_channel_content[0]):
            if make_eos:    
                communication_channel_content[0,idx] = 0
                continue
            if o==0:
                make_eos = True
        
        return communication_channel_content

    def _gen_obs_info(self, reset=False):
        if reset:
            #self.datasets["train"].datasets["train"].reset()
            self.datasets['test'].reset()
            # it is sufficient to reset the test duallabeled dataset
            # because it contains both the training and testing dataset.
            #self.datasets["test"].datasets["test"].reset()
            # But do not forget to reset classes in the train dulalabeled dataset too:
            self.datasets['train'].reset_classes()

            #print("Create dataloader: ...")
            """
            self.data_loaders = {}
            for mode, dataset in self.datasets.items():
                self.data_loaders[mode] = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.config['batch_size'],
                    shuffle=True,
                    collate_fn=collate_dict_wrapper,
                    pin_memory=True,
                    #num_workers=self.config['dataloader_num_worker']
                )
            """
            #print("Create dataloader: OK.")

            # Curriculum Distractors ?
            if self.rg_config.get('use_curriculum_nbr_distractors', False) \
            and not(hasattr(self, "init_curriculum_nbr_distractors")):
                self.init_curriculum_nbr_distractors = 1
                self.windowed_accuracy = 0.0
                self.window_count = 0
                for mode in self.datasets:
                    self.datasets[mode].setNbrDistractors(self.init_curriculum_nbr_distractors,mode=mode)
            
            self.mode = "train"

                        # Which stimulus ?
            self.stimulus_idx = 0 
            self.round_idx = 0
            self.episode_ended = False
            self.episode_ends = False 

        it_dataset = self.dataloader_index
        #data_loader = self.data_loaders[self.dataloader_index2mode[self.dataloader_index]]
        data_loader = self.datasets[self.dataloader_index2mode[self.dataloader_index]]

        if self.round_idx==0\
        and not self.episode_ended:
                self.sample = data_loader[self.stimulus_idx]
        
        # When providing feedback to listener,
        # we end the episode on the feedback timestep:
        if (self.episode_ended and self.round_idx==0)\
        or (self.episode_ends and self.round_idx==-1):
            #and self.dataloader_index>=(len(self.dataloader_index2mode)-1):
            self.done = True
        else:
            self.done = False  
        
        if self.allow_listener_query:
            speaker_observed_utterance = self.communication_history["listener"][-1]
            # need to decode it from listener viewpoint:
            speaker_observed_utterance = self.per_player_permutation[1].decode_action(
                {'communication_channel':speaker_observed_utterance}
            )["communication_channel"]
            # and encode it in the speaker viewpoint:
            speaker_observed_utterance = self.per_player_permutation[0].encode_obs(
                {'communication_channel':speaker_observed_utterance}
            )["communication_channel"]
        else:
            speaker_observed_utterance = self.communication_history["speaker"][-1]
        

        listener_observed_utterance = self.communication_history["speaker"][-1]
        # need to decode it from speaker viewpoint:
        listener_observed_utterance = self.per_player_permutation[0].decode_action(
            {'communication_channel':listener_observed_utterance}
        )["communication_channel"]
        # and encode it in the listener viewpoint:
        listener_observed_utterance = self.per_player_permutation[1].encode_obs(
            {'communication_channel':listener_observed_utterance}
        )["communication_channel"]
        
        speaker_obs = {
            "stimulus":self.sample["speaker_experiences"].reshape((-1,)),
            'communication_channel': speaker_observed_utterance,
            'other_agent_id': self.agent_ids[0],
            'role_id': self.role_ids[0],
            'mode_id': self.mode_ids[self.dataloader_index],
            'previous_game_reward': self.previous_game_reward,
            'previous_game_result': self.previous_game_result,
        }

        listener_obs = {
            "stimulus":self.sample["listener_experiences"].reshape((-1,)),
            'communication_channel': listener_observed_utterance,
            'other_agent_id': self.agent_ids[1],
            'role_id': self.role_ids[1],
            'mode_id': self.mode_ids[self.dataloader_index],
            'previous_game_reward': self.previous_game_reward,
            'previous_game_result': self.previous_game_result,
        }
        

        info = {} #{key:value for key, value in self.sample.items()}
        info["speaker_exp_latents"] = self.sample["speaker_exp_latents"].numpy()
        info["listener_exp_latents"] = self.sample["listener_exp_latents"].numpy()
        info['round_id'] = np.zeros((1,self.nbr_communication_rounds+1))
        
        if self.round_idx>=0:
            info['round_id'][0, self.round_idx] = 1

        info['nbr_communication_rounds'] = self.nbr_communication_rounds
        info['round_idx'] = self.round_idx
        info['stimulus_idx'] = self.stimulus_idx
        info['step_idx'] = self.step_count
        info['mode'] = self.dataloader_index2mode[self.dataloader_index]+f"{self.dataloader_index if self.mode=='train' else ''}"
        last_round = self.round_idx==self.nbr_communication_rounds if not(self.listener_feedback) else self.round_idx == -1 
        info['end_of_mode'] = (last_round and (self.stimulus_idx+1==len(data_loader)))
        info['nbr_successes'] = self.racc[self.dataloader_index]['nbr_successes']
        info['nbr_games'] = self.racc[self.dataloader_index]['nbr_games']
        info['running_accuracy'] = self.racc[self.dataloader_index]['nbr_successes']*100.0/(self.racc[self.dataloader_index]['nbr_games']+1e-8)
        
        
        if self.listener_feedback\
        and self.round_idx==-1\
        and not self.feedback_provided:
            listener_obs['stimulus'] = copy.deepcopy(speaker_obs['stimulus'])
            info["listener_exp_latents"] = copy.deepcopy(info["speaker_exp_latents"]) 
            self.feedback_provided = True
        else:
            self.feedback_provided = False 

        self.observations = [speaker_obs, listener_obs]
        self.infos = [copy.deepcopy(info) for _ in range(self.nbr_players)]
        
        # Bookkeeping: setting values for next call:
        self.round_idx = (self.round_idx+1)%(self.nbr_communication_rounds+1)
        
        if self.listener_feedback\
        and self.round_idx==0:
            if not self.feedback_provided:
                self.round_idx = -1
                
        if self.round_idx==0:
            self.stimulus_idx = (self.stimulus_idx+1)%len(data_loader)

            if self.stimulus_idx==0:
                self.dataloader_index = (self.dataloader_index+1)%len(self.dataloader_index2mode)
                self.mode = self.dataloader_index2mode[self.dataloader_index]
            
                if self.dataloader_index==0:
                    self.episode_ended = True
        elif self.round_idx==-1:
            if self.stimulus_idx == (len(data_loader)-1)\
            and self.dataloader_index == (len(self.dataloader_index2mode)-1):
                self.episode_ends = True
        
        if self.infos[0]['end_of_mode'] \
        and self.episode_ends:
            assert self.done

        return self.observations, self.infos

        """
        acc_keys = [k for k in logs_dict.keys() if '/referential_game_accuracy' in k]
        if len(acc_keys):
        acc = logs_dict[acc_keys[-1]].mean()

        # TODO: CURRICULUM ON DISTRATORS as a module that handles the current dataloader reference....!!
        if 'use_curriculum_nbr_distractors' in self.config\
        and self.config['use_curriculum_nbr_distractors']:
        nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
        self.stream_handler.update("signals:curriculum_nbr_distractors", nbr_distractors)
        """

        """
        # TODO: many parts everywhere, do not forget them all : CURRICULUM ON DISTRACTORS...!!!
        if 'train' in mode\
        and 'use_curriculum_nbr_distractors' in self.config\
        and self.config['use_curriculum_nbr_distractors']:
        nbr_distractors = self.datasets[mode].getNbrDistractors(mode=mode)
        windowed_accuracy = (windowed_accuracy*window_count+acc.item())
        window_count += 1
        windowed_accuracy /= window_count
        if windowed_accuracy > 75 and window_count > self.config['curriculum_distractors_window_size'] and nbr_distractors < self.config['nbr_distractors'][mode]:
        windowed_accuracy = 0
        window_count = 0
        for mode in self.datasets:
        self.datasets[mode].setNbrDistractors(self.datasets[mode].getNbrDistractors(mode=mode)+1, mode=mode)
        """

    def reset(self, **kwargs):
        self.nbr_players = 2
        self.mode = "train"
        self.done = False 

        for pidx in range(self.nbr_players):
            self.per_player_permutation[pidx].reset()

        # Step count since episode start
        self.step_count = 0

        # Communication channel:
        self.communication_history = {
            "speaker":[np.zeros((1,self.max_sentence_length))],
            "listener":[np.zeros((1,self.max_sentence_length))],
        }
        
        self.agent_ids = []
        for pidx in range(self.nbr_players):
            # random values in [0, 1) :
            pidx_ohe = self.np_random.random((1,self.id_length,))
            self.agent_ids.append(pidx_ohe)

        self.role_ids = []
        # index 0==speaker role / index 1==listener role:
        for pidx in range(2):
            pidx_ohe = np.zeros((1,2))
            pidx_ohe[0, pidx] = 1
            self.role_ids.append(pidx_ohe)

        # Which Dataloader ?
        self.dataloader_index = 0 
        #self.dataloader_index2mode = list(self.datasets.keys())
        self.dataloader_index2mode = ['train' for _ in range(self.nbr_shots)]
        self.dataloader_index2mode += ['test']
        
        self.racc = [{'nbr_successes':0, 'nbr_games':0} for _ in self.dataloader_index2mode]

        self.mode_ids = []
        # index 0==train / index 1==test:
        for mode in self.dataloader_index2mode:
            midx_ohe = np.zeros((1,2))
            if mode=='train':
                midx_ohe[0, 0] = 1
            else:
                midx_ohe[0, 1] = 1
            self.mode_ids.append(midx_ohe)

        self.previous_game_result = np.zeros((1,2))
        self.previous_game_reward = np.zeros((1,1))
        self.previous_game_was_successful = False
        
        # Return first observation
        obs, infos = self._gen_obs_info(reset=True)

        return obs, infos

    def step(self, action):
        self.step_count += 1

        self.speaker_actions = action[0]
        self.listener_actions = action[1]

        # Update communication channels:
        """
        It is safe to start by regularising the content,
        because the grounded EoS symbol (index=0) is never permutated.
        Permutation occurs only
        """
        self.speaker_actions["communication_channel"] = self._regularise_communication_channel(self.speaker_actions["communication_channel"])
        self.listener_actions["communication_channel"] = self._regularise_communication_channel(self.listener_actions["communication_channel"])
        
        self.communication_history["speaker"].append(self.speaker_actions["communication_channel"])
        self.communication_history["listener"].append(self.listener_actions["communication_channel"])
        
        self.reward = self._gen_reward()
        next_obs, next_infos = self._gen_obs_info()
        
        return next_obs, [self.reward for _ in range(self.nbr_players)], self.done, next_infos


    def _gen_reward(self):
        """
        Must imperatively be called before _gen_obs_info
        because it relies on the not-yet-updated value of round_idx.
        """
        reward = 0.0

        round_idx_reward = 0
        if self.listener_feedback:
            round_idx_reward = -1

        if self.round_idx==round_idx_reward:
            # then we have just received the listener's decision:
            if self.listener_actions["decision"] == self.sample["target_decision_idx"].item():
                self.previous_game_was_successful = True 
            else:
                self.previous_game_was_successful = False 

            if self.previous_game_was_successful:
                reward = 1.0
            else:
                if self.mode == "test":
                    reward = -2.0
                else:
                    reward = 0.0

            # accuracy bookkeeping:
            self.racc[self.dataloader_index]['nbr_games'] += 1
            self.racc[self.dataloader_index]['nbr_successes'] += int(self.previous_game_was_successful)
        
        if self.round_idx==round_idx_reward:
            self.previous_game_reward = np.ones((1,1))*reward 
            self.previous_game_result = np.zeros((1,2))
            if self.previous_game_was_successful:
                self.previous_game_result[0,1] = 1
            else:
                self.previous_game_result[0,0] = 1

        return reward

    def render(self, mode='human', close=False):
        """
        Render the whole-grid human view
        """
        font_color =  (20, 20, 255, 255)
        font_red_color =  (255, 20, 20, 255)
        font_size = 1.0 #0.5
        font = cv2.FONT_HERSHEY_TRIPLEX
        #font = cv2.FONT_HERSHEY_SIMPLEX, #font family
 
        height_px = 480
        width_px = 640
        img = 255*np.ones(shape=(height_px, width_px, 3), dtype=np.uint8)
        
        if mode == 'human':
            #self.window.show_img(img)
            #self.window.set_caption(f"Communication Channel: {self.communication_channel_content}")
            img = np.concatenate([img, 255*np.ones_like(img)], axis=1)
            orig_x = img.shape[0]
            orig_y = img.shape[1]
            
            decisions = [
                [getattr(self,'speaker_actions', {'decision':0})['decision']], 
                [getattr(self,'listener_actions', {'decision':0})['decision']],
            ]
            messages_sent = [
                getattr(self,'speaker_actions', {'communication_channel':np.zeros(self.rg_config['max_sentence_length'])})['communication_channel'].squeeze(), 
                getattr(self,'listener_actions', {'communication_channel':np.zeros(self.rg_config['max_sentence_length'])})['communication_channel'].squeeze(), 
            ]
            
            stimuli = [
                self.observations[0]['stimulus'].squeeze(),
                self.observations[1]['stimulus'].squeeze(),
            ]
            latent_stimuli = [
                self.infos[0]['speaker_exp_latents'].squeeze(),
                self.infos[0]['listener_exp_latents'].squeeze(),
            ]
            messages_received = [
                self.observations[0]['communication_channel'].squeeze(),
                self.observations[1]['communication_channel'].squeeze(),
            ]
            
            x_inc = int(orig_x*0.9)//8
            pad_x = int(orig_x*0.15)

            y_inc = int(orig_y*0.8)//2
            pad_y = int(orig_y*0.1)
            
            #print(pad_x, x_inc, pad_y, y_inc)
            
            init_x = pad_x
            pos_x = init_x

            init_y = pad_y
            for stim_idx, stim in enumerate(latent_stimuli):
                pos_y = init_y+y_inc*stim_idx
                text = 'LAT: '
                for tidx, token in enumerate(stim):
                    text += f'{int(token)} '
                position = (pos_y,pos_x)
                cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_color,
                    2,  #stroke
                )
            pos_x += x_inc
            
            init_y = pad_y
            for stim_idx, stim in enumerate(stimuli):
                pos_y = init_y+y_inc*stim_idx
                text = ''
                for tidx, token in enumerate(stim):
                    text += f'{token:.2f} '
                position = (pos_y,pos_x)
                cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_color,
                    2,  #stroke
                )
            pos_x += x_inc
            
            init_y = pad_y
            for m_idx, message in enumerate(messages_received):
                pos_y = init_y+y_inc*m_idx
                text = 'MR: '
                for tidx, token in enumerate(message):
                    text += f'{chr(97+int(token))} ' if token != 0 else 'EoS '
                position = (pos_y,pos_x)
                cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_color,
                    2,  #stroke
                )
            pos_x += int(x_inc*1.5)

            init_y = pad_y
            for didx, decision in enumerate(decisions):
                pos_y = init_y+y_inc*didx
                text = 'D: '
                for tidx, token in enumerate(decision):
                    text += f'{token} '
                position = (pos_y,pos_x)
                cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_color,
                    2,  #stroke
                )
            pos_x += x_inc
            
            init_y = pad_y
            for m_idx, message in enumerate(messages_sent):
                pos_y = init_y+y_inc*m_idx
                text = 'MS: '
                for tidx, token in enumerate(message):
                    text += f'{chr(97+int(token))} ' if token != 0 else 'EoS '
                position = (pos_y,pos_x)
                cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_color,
                    2,  #stroke
                )
            pos_x += x_inc

            # Game IDX:
            text = f"RefGame IDX: {self.mode}{self.racc[self.dataloader_index]['nbr_games']}"
            position = (int(orig_y//2), pos_x)
            cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_red_color,
                    2,  #stroke
                )
            pos_x += int(x_inc/2)
            # Result:
            acc = self.racc[self.dataloader_index]['nbr_successes']/(1.0e-3+self.racc[self.dataloader_index]['nbr_games'])*100.0 
            text = f"Accuracy : {self.racc[self.dataloader_index]['nbr_successes']}/{self.racc[self.dataloader_index]['nbr_games']} : {acc:.1f}%"
            position = (int(orig_y//2), pos_x)
            cv2.putText(
                    img,
                    text,
                    position, #position at which writing has to start
                    font,
                    font_size,
                    font_red_color,
                    2,  #stroke
                )
             
 
        if mode == 'human'\
        and getattr(self, 'window', None) == None:
            plt.imshow(img)
            plt.show()#block=False)

                  
        return img


def generate_receptive_constructive_test_env(**kwargs):
    rg_config = kwargs.get('rg_config', None)
    if rg_config is None:
        rg_config = {
            "observability":            "full",
            "max_sentence_length":      kwargs.get("max_sentence_length",3),
            "nbr_communication_rounds": kwargs.get("nbr_communication_rounds", 1),
            "nbr_distractors":          {"train":kwargs.get("nbr_distractors", 1), "test":kwargs.get("nbr_distractors", 1)},
            "distractor_sampling":      'uniform',
            # Default: use 'uniform' or "similarity-0.5"
            # otherwise the emerging language 
            # will have very high ambiguity...
            # Speakers find the strategy of uttering
            # a word that is relevant to the class/label
            # of the target, seemingly.  

            "descriptive":              kwargs.get('descriptive', False),
            "descriptive_target_ratio": 1.0/(1+kwargs.get("nbr_distractors", 1)+int(kwargs.get('descriptive', False))),

            "object_centric":           kwargs.get("nbr_object_centric_samples",1)>1,
            "nbr_stimulus":             1,

            "graphtype":                'reinforce-like',
            "tau0":                     0.2,
            "gumbel_softmax_eps":       1e-6,
            "vocab_size":               kwargs.get("vocab_size",6),
            #"force_eos":                False,
            #"symbol_embedding_size":    64, #64

            #"agent_architecture":       args.arch, #'CoordResNet18AvgPooled-2', #'BetaVAE', #'ParallelMONet', #'BetaVAE', #'CNN[-MHDPA]'/'[pretrained-]ResNet18[-MHDPA]-2'
            #"agent_learning":           "learning",  #"transfer_learning" : CNN"s outputs are detached from the graph...
            #"agent_loss_type":          args.agent_loss_type, #"NLL"

            #"cultural_pressure_it_period": args.cultural_pressure_it_period,
            #"cultural_speaker_substrate_size":  args.cultural_speaker_substrate_size,
            #"cultural_listener_substrate_size":  args.cultural_listener_substrate_size,
            #"cultural_reset_strategy":  args.cultural_reset_strategy, #"oldestL", # "uniformSL" #"meta-oldestL-SGD"
            #"cultural_reset_meta_learning_rate":  1e-3,

            # Cultural Bottleneck:
            #"iterated_learning_scheme": args.iterated_learning_scheme,
            #"iterated_learning_period": args.iterated_learning_period,
            #"iterated_learning_rehearse_MDL": args.iterated_learning_rehearse_MDL,
            #"iterated_learning_rehearse_MDL_factor": args.iterated_learning_rehearse_MDL_factor,

            # Obverter Hyperparameters:
            #"obverter_stop_threshold":  args.obverter_threshold_to_stop_message_generation,  #0.0 if not in use.
            #"obverter_nbr_games_per_round": args.obverter_nbr_games_per_round,

            #"obverter_least_effort_loss": False,
            #"obverter_least_effort_loss_weights": [1.0 for x in range(0, 10)],

            #"batch_size":               args.batch_size,
            #"dataloader_num_worker":    args.dataloader_num_worker,
            #"stimulus_depth_dim":       1 if "dSprites" in args.dataset else 3,
            #"stimulus_resize_dim":      stimulus_resize_dim, 

            #"learning_rate":            args.lr, #1e-3,
            #"adam_eps":                 1e-16,
            #"dropout_prob":             args.dropout_prob,
            #"embedding_dropout_prob":   args.emb_dropout_prob,

            #"with_gradient_clip":       False,
            #"gradient_clip":            1e0,

            #"use_homoscedastic_multitasks_loss": args.homoscedastic_multitasks_loss,

            #"use_feat_converter":       args.use_feat_converter,

            "use_curriculum_nbr_distractors": False,
            "curriculum_distractors_window_size": 25, #100,

            "unsupervised_segmentation_factor": None, #1e5
            "nbr_experience_repetition":  1,

            #"with_utterance_penalization":  False,
            #"with_utterance_promotion":     False,
            #"utterance_oov_prob":  0.5,  # Expected penalty of observing out-of-vocabulary words. 
                                                    # The greater this value, the greater the loss/cost.
            #"utterance_factor":    1e-2,

            #"with_speaker_entropy_regularization":  False,
            #"with_listener_entropy_regularization":  False,
            #"entropy_regularization_factor":    -1e-2,

            #"with_mdl_principle":       False,
            #"mdl_principle_factor":     5e-2,

            #"with_weight_maxl1_loss":   False,
        }
        kwargs['rg_config'] = rg_config
    
    # Create dataset:
    train_dataset = kwargs.get("train_dataset", None)
    if train_dataset is None:
        train_dataset = SymbolicContinuousStimulusDataset(
            train=True,
            transform=None,
            sampling_strategy=kwargs.get("sampling_strategy", None),
            split_strategy='combinatorial2-40',
            nbr_latents=kwargs.get("nbr_latents",3),
            min_nbr_values_per_latent=kwargs.get("min_nbr_values_per_latent",2),
            max_nbr_values_per_latent=kwargs.get("max_nbr_values_per_latent",5),
            nbr_object_centric_samples=kwargs.get("nbr_object_centric_samples",1),
            prototype=None,
        )

    test_dataset = SymbolicContinuousStimulusDataset(
        train=False,
        transform=None,
        sampling_strategy=kwargs.get("sampling_strategy", None),
        split_strategy='combinatorial2-40',
        nbr_latents=kwargs.get("nbr_latents",3),
        min_nbr_values_per_latent=kwargs.get("min_nbr_values_per_latent",2),
        max_nbr_values_per_latent=kwargs.get("max_nbr_values_per_latent",3),
        nbr_object_centric_samples=kwargs.get("nbr_object_centric_samples",1),
        prototype=train_dataset,
    )

    need_dict_wrapping = {}

    dataset_args = {"modes":["train", "test"]}
    dataset_args["train"] = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {
        "train": train_dataset,
        "test": test_dataset,
      },
      "need_dict_wrapping":       need_dict_wrapping,
      "nbr_stimulus":             rg_config["nbr_stimulus"],
      "distractor_sampling":      rg_config["distractor_sampling"],
      "nbr_distractors":          rg_config["nbr_distractors"],
      "observability":            rg_config["observability"],
      "object_centric":           rg_config["object_centric"],
      "descriptive":              rg_config["descriptive"],
      "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
    }
    dataset_args["test"] = {
      "dataset_class":            "DualLabeledDataset",
      "modes": {
        "train": train_dataset,
        "test": test_dataset,
      },
      "need_dict_wrapping":       need_dict_wrapping,
      "nbr_stimulus":             rg_config["nbr_stimulus"],
      "distractor_sampling":      rg_config["distractor_sampling"],
      "nbr_distractors":          rg_config["nbr_distractors"],
      "observability":            rg_config["observability"],
      "object_centric":           rg_config["object_centric"],
      "descriptive":              rg_config["descriptive"],
      "descriptive_target_ratio": rg_config["descriptive_target_ratio"],
    }

    # Create DualLabelDataset:
    using_v2 = False
    mode2dataset = dataset_args.pop('modes')
    if isinstance(mode2dataset, list):
        using_v2 = True
    
    if using_v2:
        train_dataset = dataset_args["train"]["modes"]["train"]
        need_dict_wrapping = dataset_args["train"]['need_dict_wrapping']
        if "train" in need_dict_wrapping:
            train_dataset = DictDatasetWrapper(train_dataset)
    else:
        need_dict_wrapping = dataset_args.pop('need_dict_wrapping')
        for key in need_dict_wrapping:
            mode2dataset[key] = DictDatasetWrapper(mode2dataset[key])
        
        dataset_class = dataset_args.pop('dataset_class', None)
    
        """
        if dataset_class is not None:
            Dataset = getattr(referentialgame_datasets, dataset_class)
        """
        assert dataset_class=="DualLabeledDataset"

    rg_datasets = {}
    for mode in mode2dataset:
        if using_v2:
            dataset = dataset_args[mode].pop("modes")[mode]
            need_dict_wrapping = dataset_args[mode].pop('need_dict_wrapping')
            if mode in need_dict_wrapping:
                dataset = DictDatasetWrapper(dataset)
            
            dataset_class = dataset_args[mode].pop('dataset_class', None)
            if dataset_class is not None:
                Dataset = DualLabeledDataset
                #Dataset = getattr(referentialgame_datasets, dataset_class)    
        else:
            dataset = mode2dataset[mode]

        ###

        if Dataset is None:
            rg_datasets[mode] = dataset
        else:
            if using_v2:
                inner_dataset_args = copy.deepcopy(dataset_args[mode])
            else:
                inner_dataset_args = copy.deepcopy(dataset_args)
            
            if dataset_class == 'LabeledDataset': 
                inner_dataset_args['dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)
            elif dataset_class == 'DualLabeledDataset':
                if using_v2:
                    inner_dataset_args['train_dataset'] = train_dataset
                else:
                    inner_dataset_args['train_dataset'] = mode2dataset["train"]
                inner_dataset_args['test_dataset'] = dataset
                inner_dataset_args['mode'] = mode
                rg_datasets[mode] = Dataset(kwargs=inner_dataset_args)

    kwargs['datasets'] = rg_datasets
    rg_config["nbr_distractors"] = rg_config["nbr_distractors"]['train']
    rg_config["nbr_latents"] = kwargs.get("nbr_latents",3)

    env = SymbolicBehaviourBenchmark_ReceptiveConstructiveTestEnv(**kwargs)

    return env 

def generate_receptive_constructive_test_env_2shots(**kwargs):
    kwargs['nbr_shots'] = 2
    return generate_receptive_constructive_test_env(**kwargs) 
