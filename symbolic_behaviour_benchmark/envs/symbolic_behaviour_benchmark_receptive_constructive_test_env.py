from typing import Dict, List, Optional 

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2

import gym
from gym import spaces
from gym.utils import seeding

import re
import logging
import copy

logger = logging.getLogger(__name__)

from symbolic_behaviour_benchmark.envs.communication_channel import CommunicationChannel 
from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import SymbolicContinuousStimulusDataset 

from symbolic_behaviour_benchmark.utils import DualLabeledDataset
from symbolic_behaviour_benchmark.utils import DictDatasetWrapper

from symbolic_behaviour_benchmark.utils.pybullet_renderer	import PyBulletRenderer
from symbolic_behaviour_benchmark.utils.utils import STR2BT, BT2STR


def scs_to_image(scs_values):
    N_dim = len(scs_values)  # Number of attribute/factor dimensions
    fig, ax = plt.subplots(N_dim, 1, figsize=(6, N_dim * 2))

    for i, value in enumerate(scs_values):
        ax[i].set_xlim(-1, 1)
        ax[i].set_ylim(-0.5, 0.5)
        ax[i].axis('off')

        # Create a circle at the position corresponding to the scs value
        circle = Circle((value, 0), radius=0.1, color='blue')
        ax[i].add_patch(circle)

    plt.subplots_adjust(hspace=0)
    plt.axis('off')
    # Convert the plot to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img

class CommunicationChannelPermutation(object):
    def __init__(self, env, identity=False):
        self.env = env 
        self.identity = identity

        self.vocab_size = self.env.vocab_size-1 # in order to account for EoS
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
        max_prompt_sentence_length=2**16,
        include_prompts=False,
        floating_point_precision=3,
        discussion_mode=False,
        **kwargs,
    ):  
        super(SymbolicBehaviourBenchmark_ReceptiveConstructiveTestEnv, self).__init__()
        self.kwargs = kwargs
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
        self.max_prompt_sentence_length = max_prompt_sentence_length
        self.include_prompts = include_prompts
        self.floating_point_precision = floating_point_precision
        self.discussion_mode = discussion_mode

        # 3D Renderer:
        self.renderer = None
        if self.kwargs.get('domain', 'SCS') == '3D':
            self.renderer = PyBulletRenderer(N_dim=self.nbr_latents)

        # Categorical domain: text-conversion callable for prompt builders.
        self._stimulus_to_text = None
        if self.kwargs.get('domain', 'SCS') == 'categorical':
            base_ds = datasets['train'].datasets['train']
            if not hasattr(base_ds, 'latent_class_to_text'):
                raise TypeError(
                    f"domain='categorical' requires a CategoricalStimulusDataset at "
                    f"datasets['train'].datasets['train'], got {type(base_ds).__name__}"
                )
            self._stimulus_to_text = base_ds.latent_class_to_text

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
        _stim_low = 0 if self.kwargs.get('domain', 'SCS') == 'categorical' else -1
        _stim_high = (self.kwargs.get('max_nbr_values_per_latent', 5) - 1
                      if self.kwargs.get('domain', 'SCS') == 'categorical' else 1)
        self.stimulus_observation_space = spaces.Box(
            low=_stim_low,
            high=_stim_high,
            shape=((self.nbr_distractors+1)*rg_config.get('nbr_stimulus', 1)*self.nbr_latents, ),
            dtype=np.float32
        )
        self.communication_channel_observation_space = copy.deepcopy(self.communication_channel_action_space)

        self.id_length = 3
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
            CommunicationChannelPermutation(
                env=self, 
                identity=not(self.use_communication_channel_permutations),
            )
            for _ in range(self.nbr_players)
        ]

        self.seed(seed)

    def _update_speaker_prompt(self, obs, info, context_prompt=""):
        ''' 
        Update the prompt based on the current observation.
        TODO: update to multiple distractor stimuli...
        '''
        assert self.nbr_distractors == 0
        printoptions = np.get_printoptions()
        np.set_printoptions(formatter={'float_kind': lambda x: f"%.{self.floating_point_precision}f" % x})

        round_idx_reward = 0
        if self.listener_feedback:
            round_idx_reward = -1

        game_id = info.get('cumulative_stimulus_idx', info['stimulus_idx'])
        step_id = info['round_idx']
        
        if context_prompt == "": 
          context_prompt = f"You and your partner are playing a sequence of referential games. "
          context_prompt += f"You are the speaker.\n"
          # TODO: context_prompt += f"Your partner has id {obs['other_agent_id']}.\n"
          
          context_prompt += f"In the first phase, you will get acquainted with "
          context_prompt += f"the atomic components of the possible observations. "
          context_prompt += f"Then, you will be tested with "
          context_prompt += f"new observations, combining the same atomic components in novel ways.\n"
          
          context_prompt += f"At each game, each of you observes a stimulus, "
          context_prompt += f"which represents a latent meaning, "
          context_prompt += f"and your common goal is to figure out whether you are observing "
          context_prompt += f"different or similar latent meanings. "
          context_prompt += f"You can communicate with your partner using the communication channel. "
          context_prompt += f"The communication channel is made up of {self.vocab_size} symbols "
          context_prompt += f"that you can combine together to form a sentence of "
          context_prompt += f"maximum length {self.max_sentence_length}. "
          context_prompt += f"Beware that symbol 0 is grounded already. "
          context_prompt += f"It is the end-of-message symbol. "
          context_prompt += f"It means that any symbol that comes after it will be ignored "
          context_prompt += f"and regularised into symbol 0.\n"
        
          context_prompt += f"From one game to the next, you should aim to be consistent "
          context_prompt += f"so that your partner can figure out the code that you are using "
          context_prompt += f"to communicate and decrypt messages towards "
          context_prompt += f"fulfilling your common goal.\n"
        else:
          context_prompt = context_prompt.replace("are observing", "have observed")
          context_prompt = context_prompt.replace("here is a ", "there was a ")
          context_prompt = context_prompt.replace("partner is being", "partner was being")
          context_prompt = context_prompt.replace("you observe.", "you observed.")
          #context_prompt = context_prompt.replace("It is an ", "It was an ")
          # Provide results about the previous game:
          if step_id == round_idx_reward:
            context_prompt += f"Your partner has decided that both of you were observing "
            if self.listener_actions["decision"] > 0 :# TODO == self.sample["target_decision_idx"].item():
              context_prompt += "different latent meanings.\n"
            else:
              context_prompt += "similar latent meanings.\n"
            successful_game = self.listener_actions["decision"] == self.sample["target_decision_idx"].item()
            if successful_game:
              context_prompt += f"This was correct. " 
            else:
              context_prompt += f"This was incorrect. " 
            context_prompt += f"You and your partner have "
            context_prompt += f"{'won' if successful_game else'lost'} "
            context_prompt += f"game #{game_id}.\n"

        comm_channel_char = obs['communication_channel'][0].astype(int).tolist()
        # For grounding purposes, we need to keep integers.
        #comm_channel_char = [chr(i) for i in obs['communication_channel'][0].astype(int).tolist()]
        
        # Previous message from the speaker:
        prev_speaker_utterance = self.communication_history["speaker"][-1]
        # no need to decode it from speaker viewpoint:
        #prev_speaker_utterance = self.per_player_permutation[0].decode_action(
        #    {'communication_channel':prev_speaker_utterance}
        #)["communication_channel"]
        #prev_comm_channel_char = prev_speaker_utterance[0].astype(int).tolist()
        prev_comm_channel_char = prev_speaker_utterance.astype(int).tolist()

        if step_id == 0:
          _stim = obs['stimulus'].reshape(-1).numpy()
          _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
          context_prompt += f"\nStarting game #{game_id}, this is the new stimulus: "
          context_prompt += f"{_stim_repr}.\n"
        elif step_id != -1:
          context_prompt = context_prompt.replace(
            f"\nStarting game #{game_id}, this is the new stimulus: ",
            f"\nAt game #{game_id}, you are observing stimulus: ",
          )
          #context_prompt += f"\nAt game #{game_id}, step #{step_id}, you are observing the "
          #context_prompt += f"following stimulus: {obs['stimulus'].reshape(-1).numpy()}.\n"
          if step_id != 0:
            context_prompt += f"You have sent the following message: {prev_comm_channel_char}.\n"
            if self.allow_listener_query:
              context_prompt += f"Your partner has sent you the following message: {comm_channel_char}.\n"
        else:
          # No update:
          #context_prompt += "\n"
          context_prompt += f"\nAt the end of game #{game_id}, here is a special step "
          context_prompt += f"where your partner is being shown the exact stimulus that you "
          context_prompt += f"observe.\n"
          #context_prompt += f" It is an opportunity for them to sync with you by verifying "
          #context_prompt += f"that they understood your message.\n"
        
        self.speaker_context_prompt = context_prompt

        question_prompt = f"\nYou are an expert in the matter. Given the information above, answer the following question(s) to the best of your abilities.\n\n"

        question_prompt += f"Question #1: Do you think your partner understands your messages?\n"
        question_prompt += f"Answer either 0.:'Yes' or 1.:'No'.\n\n"

        question_prompt += f"Question #2: What message should you send to your partner to better "
        question_prompt += f"coordinate together towards fulfilling your common goal?\n"
        question_prompt += f"The message is made up of {self.max_sentence_length} symbols, "
        question_prompt += f"each of which can be filled with one of the {self.vocab_size} "
        question_prompt += f"vocabulary symbols. For example: "
        question_prompt += f"{self.communication_channel_action_space.sample()[0].tolist()}.\n"
        question_prompt += f"This question corresponds to {self.max_sentence_length} implicit "
        question_prompt += f"questions, one for each of the {self.max_sentence_length} symbols "
        question_prompt += f"of the message. Thus, each possible answer id is between 0 and {self.vocab_size-1}, corresponding to one of the {self.vocab_size} vocabulary symbols.\n"
         
        speaker_prompt = context_prompt+question_prompt
        
        # Eventhough we only ask two questions, we want to retrieve a message of length 
        # max_sentence_length in the second one, where each positions can be filled with 
        # one of the vocab symbols.
        speaker_prompt += f"\n[NBR_QUESTIONS]{self.max_sentence_length+1}[/NBR_QUESTIONS]\n"
        speaker_prompt += f"[MAX_NBR_OPTIONS]{max(2,self.vocab_size)}[/MAX_NBR_OPTIONS]\n"

        bt_speaker_prompt = STR2BT(speaker_prompt, max_sentence_length=self.max_prompt_sentence_length)

        # ── Discussion mode: also emit per-step and intro prompts ─────────────
        bt_speaker_step_prompt = None
        bt_speaker_intro_prompt = None
        if self.discussion_mode:
            if not self._speaker_intro_text:
                intro_end = self.speaker_context_prompt.find("\nStarting game")
                if intro_end == -1:
                    intro_end = len(self.speaker_context_prompt)
                self._speaker_intro_text = self.speaker_context_prompt[:intro_end].strip()

            step_text = ""
            if self._speaker_pending_feedback:
                step_text += self._speaker_pending_feedback
                self._speaker_pending_feedback = ""

            if step_id == 0:
                _stim = obs['stimulus'].reshape(-1).numpy()
                _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
                step_text += (f"\nStarting game #{game_id}, this is the new stimulus: "
                              f"{_stim_repr}.\n")
            elif step_id != -1:
                _stim = obs['stimulus'].reshape(-1).numpy()
                _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
                step_text += (f"\nAt game #{game_id}, you are observing stimulus: "
                              f"{_stim_repr}.\n")
                step_text += f"You have sent the following message: {prev_comm_channel_char}.\n"
                if self.allow_listener_query:
                    step_text += (f"Your partner has sent you the following message: "
                                  f"{comm_channel_char}.\n")
            else:
                step_text += (f"\nAt the end of game #{game_id}, here is a special step "
                              f"where your partner is being shown the exact stimulus that you "
                              f"observe.\n")

            if step_id == round_idx_reward and hasattr(self, 'listener_actions'):
                fb = "Your partner has decided that both of you were observing "
                fb += ("different latent meanings.\n"
                       if self.listener_actions["decision"] > 0
                       else "similar latent meanings.\n")
                successful_game = (self.listener_actions["decision"]
                                   == self.sample["target_decision_idx"].item())
                fb += "This was correct. " if successful_game else "This was incorrect. "
                fb += f"You and your partner have {'won' if successful_game else 'lost'} game #{game_id}.\n"
                self._speaker_pending_feedback = fb

            speaker_step_prompt_text = step_text + question_prompt
            speaker_step_prompt_text += f"\n[NBR_QUESTIONS]{self.max_sentence_length+1}[/NBR_QUESTIONS]\n"
            speaker_step_prompt_text += f"[MAX_NBR_OPTIONS]{max(2,self.vocab_size)}[/MAX_NBR_OPTIONS]\n"
            bt_speaker_step_prompt = STR2BT(
                speaker_step_prompt_text, max_sentence_length=self.max_prompt_sentence_length
            )
            bt_speaker_intro_prompt = STR2BT(
                self._speaker_intro_text, max_sentence_length=self.max_prompt_sentence_length
            )

        np.set_printoptions(**printoptions)
        return bt_speaker_prompt, speaker_prompt, bt_speaker_step_prompt, bt_speaker_intro_prompt

    def _update_listener_prompt(self, obs, info, context_prompt=""):
        ''' 
        Update the prompt based on the current observation.
        TODO: update to multiple distractor stimuli and allow listener query...
        '''
        assert self.nbr_distractors == 0
        assert not self.allow_listener_query
        printoptions = np.get_printoptions()
        np.set_printoptions(formatter={'float_kind': lambda x: f"%.{self.floating_point_precision}f" % x})

        round_idx_reward = 0
        if self.listener_feedback:
            round_idx_reward = -1

        game_id = info.get('cumulative_stimulus_idx', info['stimulus_idx'])
        step_id = info['round_idx']
 
        if context_prompt == "": 
          context_prompt = f"You and your partner are playing a sequence of referential games. "
          context_prompt += f"You are the listener.\n"
          # TODO: context_prompt += f"Your partner has id {obs['other_agent_id']}.\n"
          
          context_prompt += f"In the first phase, you will get acquainted with "
          context_prompt += f"the atomic components of the possible observations. "
          context_prompt += f"Then, you will be tested with "
          context_prompt += f"new observations, combining the same atomic components in novel ways.\n"
          
          context_prompt += f"At each game, each of you observes a stimulus, "
          context_prompt += f"which represents a latent meaning, "
          context_prompt += f"and your common goal is to figure out whether you are observing "
          context_prompt += f"different or similar latent meanings. "
          context_prompt += f"To help you do so, your partner can send you messages using the "
          context_prompt += f"communication channel, which is made up of {self.vocab_size} symbols "
          context_prompt += f"that can be combined together to form a sentence of maximum length "
          context_prompt += f"{self.max_sentence_length}.\n"
          context_prompt += f"Beware that symbol 0 is grounded already. "
          context_prompt += f"It is the end-of-message symbol. "
          context_prompt += f"It means that any symbol that comes after it will be ignored "
          context_prompt += f"and regularised into symbol 0.\n"
        else:
          context_prompt = context_prompt.replace("are observing", "have observed")
          context_prompt = context_prompt.replace("partner has sent", "partner had sent")
          context_prompt = context_prompt.replace(
            "this is the exact stimulus that your partner was observing",
            "this was the exact stimulus that your partner was observing",
          )
          '''
          context_prompt = context_prompt.replace("partner observes", "partner had observed")
          context_prompt = context_prompt.replace("here is a ", "there was a ")
          context_prompt = context_prompt.replace("this is the exact ", "this was the exact ")
          context_prompt = context_prompt.replace("you are given", "you were given")
          '''
          context_prompt = re.sub(
            r'\nAt the end of game #(\d+), here is a special step '
            r'where you are given an opportunity to sync with your partner: '
            r'this is the exact stimulus that your partner observes',
            lambda m: f'\nIn game #{m.group(1)}, this is the exact stimulus that your partner was observing',
            context_prompt,
          )
          # Provide results about the previous game:
          if step_id == round_idx_reward:
            context_prompt += f"You have decided that both of you were observing "
            if self.listener_actions["decision"] > 0 :# TODO == self.sample["target_decision_idx"].item():
              context_prompt += "different latent meanings.\n"
            else:
              context_prompt += "similar latent meanings.\n"
            successful_game = self.listener_actions["decision"] == self.sample["target_decision_idx"].item()
            if successful_game:
              context_prompt += f"This was correct. " 
            else:
              context_prompt += f"This was incorrect. " 
            context_prompt += f"You and your partner have "
            context_prompt += f"{'won' if successful_game else'lost'} "
            context_prompt += f"game #{game_id}.\n"

        if step_id == 0:
          _stim = obs['stimulus'].reshape(-1).numpy()
          _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
          context_prompt += f"\nStarting game #{game_id}, this is the new stimulus: "
          context_prompt += f"{_stim_repr}.\n"
        elif step_id != -1:
          context_prompt = context_prompt.replace(
            f"\nStarting game #{game_id}, this is the new stimulus: ",
            f"\nAt game #{game_id}, you are observing stimulus: ",
          )
          #context_prompt += f"\nAt game #{game_id}, step #{step_id}, you are observing the "
          #context_prompt += f"following stimulus: {obs['stimulus'].reshape(-1).numpy()}.\n"
        else:
          _stim = obs['stimulus'].reshape(-1).numpy()
          _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
          context_prompt += f"\nAt the end of game #{game_id}, here is a special step "
          context_prompt += f"where you are given an opportunity to sync with your partner: "
          context_prompt += f"this is the exact stimulus that "
          context_prompt += f"your partner observes: {_stim_repr}.\n"

        comm_channel_char = obs['communication_channel'][0].astype(int).tolist()
        #comm_channel_char = [chr(i) for i in obs['communication_channel'][0].astype(int).tolist()]
        
        if step_id != 0 \
        and step_id != -1:
          context_prompt += f"Your partner has sent you the following message: "
          context_prompt += f"{comm_channel_char}.\n"

        self.listener_context_prompt = context_prompt
        
        question_prompt = f"\nYou are an expert in the matter. Given the information above, answer the following question(s) to the best of your abilities.\n\n"

        question_prompt += f"Question #1: At the current game #{game_id}, do think that you are observing a stimulus representing the same latent meaning as the stimulus that your partner is observing?\n"
        question_prompt += f"Answer either 0.:'Yes' or 1.:'No'.\n\n"

        if self.allow_listener_query:
            question_prompt += f"Question #2: What message should you send your partner "
            question_prompt += f"to better coordinate with them towards fulfilling your common goal?\n"
            question_prompt += f"The message is made up of {self.max_sentence_length} symbols, "
            question_prompt += f"each of which can be filled with one of the {self.vocab_size} "
            question_prompt += f"vocabulary symbols. For example: "
            question_prompt += f"{self.communication_channel_action_space.sample()[0].tolist()}.\n"
            question_prompt += f"This question corresponds to {self.max_sentence_length} implicit "
            question_prompt += f"questions, one for each of the {self.max_sentence_length} symbols "
            question_prompt += f"of the message. Thus, each possible answer id is between 0 and {self.vocab_size-1}, corresponding to one of the {self.vocab_size} vocabulary symbols.\n"

        listener_prompt = context_prompt+question_prompt

        _nbr_questions = (self.max_sentence_length + 1) if self.allow_listener_query else 1
        _max_options = max(2, self.vocab_size) if self.allow_listener_query else max(2, self.nbr_distractors + 1 + int(self.rg_config.get('descriptive', True)))
        listener_prompt += f"\n[NBR_QUESTIONS]{_nbr_questions}[/NBR_QUESTIONS]\n"
        listener_prompt += f"[MAX_NBR_OPTIONS]{_max_options}[/MAX_NBR_OPTIONS]\n"

        bt_listener_prompt = STR2BT(listener_prompt, max_sentence_length=self.max_prompt_sentence_length)

        # ── Discussion mode: also emit per-step and intro prompts ─────────────
        bt_listener_step_prompt = None
        bt_listener_intro_prompt = None
        if self.discussion_mode:
            # Capture static intro once (set on first call when context_prompt was "").
            if not self._listener_intro_text:
                intro_end = self.listener_context_prompt.find("\nStarting game")
                if intro_end == -1:
                    intro_end = len(self.listener_context_prompt)
                self._listener_intro_text = self.listener_context_prompt[:intro_end].strip()

            # Build per-step text: pending feedback + current stimulus/message.
            # Only consume pending feedback at the listener's action round so the
            # LLM actually sees it (step_id=0 is the speaker round; nobody reads it).
            step_text = ""
            if self._listener_pending_feedback and step_id == self.nbr_communication_rounds:
                step_text += self._listener_pending_feedback
                self._listener_pending_feedback = ""

            if step_id == 0:
                _stim = obs['stimulus'].reshape(-1).numpy()
                _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
                step_text += (f"\nStarting game #{game_id}, this is the new stimulus: "
                              f"{_stim_repr}.\n")
            elif step_id != -1:
                _stim = obs['stimulus'].reshape(-1).numpy()
                _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
                step_text += (f"\nAt game #{game_id}, you are observing stimulus: "
                              f"{_stim_repr}.\n")
                step_text += (f"Your partner has sent you the following message: "
                              f"{comm_channel_char}.\n")
            else:
                _stim = obs['stimulus'].reshape(-1).numpy()
                _stim_repr = str(self._stimulus_to_text(_stim)) if self._stimulus_to_text else str(_stim)
                step_text += (f"\nAt the end of game #{game_id}, here is a special step "
                              f"where you are given an opportunity to sync with your partner: "
                              f"this is the exact stimulus that your partner observes: "
                              f"{_stim_repr}.\n")

            # Stash feedback text for prepending to the NEXT user turn.
            # Guard with hasattr to avoid accessing listener_actions before the
            # first env.step() has been called (listener_actions is set in step()).
            if step_id == round_idx_reward and hasattr(self, 'listener_actions'):
                successful_game = (self.listener_actions["decision"]
                                   == self.sample["target_decision_idx"].item())
                result_text = "You have decided that both of you were observing "
                result_text += ("different latent meanings.\n"
                                if self.listener_actions["decision"] > 0
                                else "similar latent meanings.\n")
                result_text += "This was correct. " if successful_game else "This was incorrect. "
                result_text += f"You and your partner have {'won' if successful_game else 'lost'} game #{game_id}.\n"
                if self.listener_feedback:
                    # Listener-feedback is on: step_text here IS the feedback round
                    # content (listener's stimulus replaced with speaker's exact stimulus).
                    # Carry the full step_text so the LLM sees the speaker's actual stimulus.
                    self._listener_pending_feedback = step_text + result_text
                else:
                    # Listener-feedback is off: no speaker-stimulus reveal; carry result only.
                    self._listener_pending_feedback = result_text

            if self.allow_listener_query:
                _format_reminder = (
                    f"\nIMPORTANT: Respond with ONLY {self.max_sentence_length + 1} "
                    f"space-separated integers on a single line "
                    f"(decision token1 token2 token3). "
                    f"Example: 0 2 3 1. "
                    f"Your response has a strict token budget — do NOT include any explanation or reasoning. "
                    f"If your response is too long or gets cut off, you will be re-prompted and must answer more concisely.\n"
                )
            else:
                _format_reminder = (
                    f"\nIMPORTANT: Respond with ONLY 1 integer (your decision: 0 for same, 1 for different). "
                    f"Example: 0. "
                    f"Your response has a strict token budget — do NOT include any explanation or reasoning. "
                    f"If your response is too long or gets cut off, you will be re-prompted and must answer more concisely.\n"
                )
            listener_step_prompt_text = step_text + question_prompt + _format_reminder
            listener_step_prompt_text += f"\n[NBR_QUESTIONS]{_nbr_questions}[/NBR_QUESTIONS]\n"
            listener_step_prompt_text += f"[MAX_NBR_OPTIONS]{_max_options}[/MAX_NBR_OPTIONS]\n"
            bt_listener_step_prompt = STR2BT(
                listener_step_prompt_text, max_sentence_length=self.max_prompt_sentence_length
            )
            bt_listener_intro_prompt = STR2BT(
                self._listener_intro_text, max_sentence_length=self.max_prompt_sentence_length
            )

        np.set_printoptions(**printoptions)
        return bt_listener_prompt, listener_prompt, bt_listener_step_prompt, bt_listener_intro_prompt

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
            # Context prompts:
            self.speaker_context_prompt = ""
            self.listener_context_prompt = ""
            # Discussion-mode state (no-op when discussion_mode=False):
            self._listener_intro_text = ""
            self._listener_pending_feedback = ""
            self._speaker_intro_text = ""
            self._speaker_pending_feedback = ""

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
            self._cumulative_stimulus_idx = 0
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
        info['cumulative_stimulus_idx'] = self._cumulative_stimulus_idx
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

        # Speaker and Listener prompts:
        if self.include_prompts:
            (self.bt_speaker_prompt, self.speaker_prompt,
             self.bt_speaker_step_prompt, self.bt_speaker_intro_prompt) = \
                self._update_speaker_prompt(
                    obs=speaker_obs,
                    info=info,
                    context_prompt=self.speaker_context_prompt,
                )
            (self.bt_listener_prompt, self.listener_prompt,
             self.bt_listener_step_prompt, self.bt_listener_intro_prompt) = \
                self._update_listener_prompt(
                    obs=listener_obs,
                    info=info,
                    context_prompt=self.listener_context_prompt,
                )

        self.observations = [speaker_obs, listener_obs]
        self.infos = [copy.deepcopy(info) for _ in range(self.nbr_players)]
        if self.include_prompts:
            self.infos[0]["prompt"] = self.bt_speaker_prompt
            self.infos[1]["prompt"] = self.bt_listener_prompt
            if self.discussion_mode:
                if self.bt_speaker_step_prompt is not None:
                    self.infos[0]["step_prompt"] = self.bt_speaker_step_prompt
                    self.infos[0]["intro_prompt"] = self.bt_speaker_intro_prompt
                if self.bt_listener_step_prompt is not None:
                    self.infos[1]["step_prompt"] = self.bt_listener_step_prompt
                    self.infos[1]["intro_prompt"] = self.bt_listener_intro_prompt
         
        # Bookkeeping: setting values for next call:
        self.round_idx = (self.round_idx+1)%(self.nbr_communication_rounds+1)
        
        if self.listener_feedback\
        and self.round_idx==0:
            if not self.feedback_provided:
                self.round_idx = -1
                
        if self.round_idx==0:
            self.stimulus_idx = (self.stimulus_idx+1)%len(data_loader)
            self._cumulative_stimulus_idx += 1

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

        # convert to image:
        if self.kwargs.get('domain', 'SCS') =='2D':
          obs_image = [copy.deepcopy(o) for o in obs]
          for pidx in range(self.nbr_players): 
            obs_image[pidx]["stimulus_2D"] = scs_to_image(obs[pidx]['stimulus'])
          obs = obs_image
        elif self.kwargs.get('domain', 'SCS') =='3D':
          obs_image = [copy.deepcopy(o) for o in obs]
          for pidx in range(self.nbr_players): 
            obs_image[pidx]["stimulus_3D"] = self.renderer.render(obs[pidx]['stimulus'])
          obs = obs_image
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
        
        # convert to image:
        if self.kwargs.get('domain', 'SCS') =='2D':
          obs_image = [copy.deepcopy(o) for o in next_obs]
          for pidx in range(self.nbr_players): 
            obs_image[pidx]["stimulus_2D"] = scs_to_image(next_obs[pidx]['stimulus'])
          next_obs = obs_image
        elif self.kwargs.get('domain', 'SCS') =='3D':
          obs_image = [copy.deepcopy(o) for o in next_obs]
          for pidx in range(self.nbr_players): 
            obs_image[pidx]["stimulus_3D"] = self.renderer.render(next_obs[pidx]['stimulus'])
          next_obs = obs_image
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
    if kwargs.get('domain', 'SCS') == 'categorical':
        from symbolic_behaviour_benchmark.categorical_stimulus_dataset import (
            CategoricalStimulusDataset,
        )
        _DatasetClass = CategoricalStimulusDataset
    else:
        _DatasetClass = SymbolicContinuousStimulusDataset

    train_dataset = kwargs.get("train_dataset", None)
    if train_dataset is None:
        train_dataset = _DatasetClass(
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

    test_dataset = _DatasetClass(
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
