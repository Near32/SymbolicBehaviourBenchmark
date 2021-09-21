from typing import Dict, List, Optional 

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import logging 
import copy

logger = logging.getLogger(__name__)

from symbolic_behaviour_benchmark.symbolic_continuous_stimulus_dataset import SymbolicContinuousStimulusDataset 

from symbolic_behaviour_benchmark.utils import DualLabeledDataset
from symbolic_behaviour_benchmark.utils import DictDatasetWrapper


class SymbolicBehaviourBenchmark_RecallTestEnv(gym.Env):
    """
    """
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        rg_config:Dict[str,str],
        datasets:Dict[str,DualLabeledDataset],
        max_episode_length=None,
        min_nbr_values_per_latent=2,
        max_nbr_values_per_latent=10,
        representation_type="ohe",
        seed=1337,
        nbr_shots=1,
        **kwargs,
        ):  
        """
        This environment aims to test the agents ability to recall the index of the value instantiated
        on a prompted attribute/latent dimension of a past stimulus, following a presentation
        of an ordered list of stimuli.

        The task can be carried out with a one-hot encoding embedding (thus making use of
        exactly :param max_nbr_values_per_latent: different values on each latent dimension)
        or with the symbolic continuous representation (where the number of values for each latent
        is randomly choosen from one episode to the other within the given range). 
        The symbolic continuous representation poses an extra problem to the agent in the form 
        of a (variable) binding problem: 
        on top of having to learning how to store the stimuli to be able to later conveniently
        recall them, the agent must learn to figure out how many values are there on each latent 
        dimension, before being able to recall the discrete index of the value that was instantiated
        in the recalled stimulus.
        """
        super(SymbolicBehaviourBenchmark_RecallTestEnv, self).__init__()
        
        self.max_episode_length = max_episode_length
        self.min_nbr_values_per_latent = min_nbr_values_per_latent
        self.max_nbr_values_per_latent = max_nbr_values_per_latent
        self.representation_type = representation_type

        if self.representation_type == "ohe":
            self.min_nbr_values_per_latent = self.max_nbr_values_per_latent
        else:
            assert self.representation_type == "scr"

        self.rg_config = rg_config
        self.datasets = datasets
        assert 'train' in self.datasets.keys()
        assert 'test' in self.datasets.keys()
        self.mode = 'train'
        
        self.nbr_latents = rg_config.get("nbr_latents", None)
        self.nbr_shots = nbr_shots 
        self.max_mode_length = self.max_episode_length // self.nbr_shots if self.max_episode_length is not None else None

        # Actions consist of a decision that is discrete integer valued:
        # The last action corresponds to NO-OP.
        self.action_space = spaces.Discrete(self.max_nbr_values_per_latent+1)
       
        # Observations are dictionaries containing:
        # -stimulus,
        # -previous game's success boolean,
        self.stimulus_shape = (rg_config.get('nbr_stimulus', 1)*self.nbr_latents, ) 
        if self.representation_type=="ohe":
            self.stimulus_shape = (self.stimulus_shape[0]*self.max_nbr_values_per_latent,)

        self.stimulus_observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=self.stimulus_shape,
            dtype=np.float32
        )

        self.previous_game_result_observation_space = spaces.MultiBinary(n=2)
        # toggle index 0==failure / 1==success

        self.observation_space = spaces.Dict({
            'stimulus': self.stimulus_observation_space,
            'previous_game_result': self.previous_game_result_observation_space,
        })
        
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed 

    def _gen_obs_info(self, reset=False):
        if reset:
            #self.datasets["train"].datasets["train"].reset()
            self.datasets['test'].reset()
            # it is sufficient to reset the test duallabeled dataset
            # because it contains both the training and testing dataset.
            #self.datasets["test"].datasets["test"].reset()
            # But do not forget to reset classes in the train dulalabeled dataset too:
            self.datasets['train'].reset_classes()
            
            self.dataloader_index2mode_length = [
                min(
                    self.max_mode_length if self.max_mode_length is not None else 1e6,
                    len(self.datasets[mode]),
                )
                for mode in self.dataloader_index2mode
            ]
            self.dataloader_index2indices = [
                self.np_random.choice(
                    range(len(self.datasets[mode])),
                    self.dataloader_index2mode_length[idx],
                )
                for idx, mode in enumerate(self.dataloader_index2mode)
            ]
            
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

            self.mode = "train"

            self.stimulus_idx = 0 
            self.round_idx = 0
            self.episode_ended = False

        it_dataset = self.dataloader_index
        #data_loader = self.data_loaders[self.dataloader_index2mode[self.dataloader_index]]
        data_loader = self.datasets[self.dataloader_index2mode[self.dataloader_index]]

        if self.round_idx==0\
        and not self.episode_ended:
                self.sample = data_loader[
                    self.dataloader_index2indices[self.dataloader_index][self.stimulus_idx]
                ]

        if self.episode_ended \
        and self.round_idx==0 \
        and self.dataloader_index>=(len(self.dataloader_index2mode)-1):
            self.done = True
        else:
            self.done = False  
        
        if self.round_idx==0:
            """
            We show the stimulus:
            """
            stimulus = self.sample["speaker_experiences"].reshape((-1,)) 
            if self.representation_type=="ohe":
                stimulus = self.sample["speaker_exp_latents_one_hot_encoded"].reshape((-1,))
        else:
            """
            We show a vector indicating the queried attribute/latent dimension:
            """
            self.queried_attr_idx = self.np_random.randint(self.nbr_latents)

            stimulus = np.zeros(self.stimulus_shape).reshape((-1,))
            if self.representation_type=="ohe":
                start = self.max_nbr_values_per_latent*self.queried_attr_idx
                end = start+self.max_nbr_values_per_latent
                stimulus[start:end] = 1
            else:
                stimulus[self.queried_attr_idx] = 1.0

        obs = {
            "stimulus":stimulus,
            'previous_game_result': self.previous_game_result,
        }

        self.observations = [obs]

        info = {} #{key:value for key, value in self.sample.items()}
        info["exp_latents"] = self.sample["speaker_exp_latents"].numpy()
        info['round_id'] = np.zeros((1,2))
        info['round_id'][0, self.round_idx] = 1

        info['round_idx'] = self.round_idx
        info['stimulus_idx'] = self.stimulus_idx
        info['step_idx'] = self.step_count
        info['mode'] = self.dataloader_index2mode[self.dataloader_index]+f"{self.dataloader_index if self.mode=='train' else ''}"
        info['end_of_mode'] = (self.round_idx==1 and (self.stimulus_idx+1==len(data_loader)))
        info['nbr_successes'] = self.racc[self.dataloader_index]['nbr_successes']
        info['nbr_games'] = self.racc[self.dataloader_index]['nbr_games']
        info['running_accuracy'] = self.racc[self.dataloader_index]['nbr_successes']*100.0/(self.racc[self.dataloader_index]['nbr_games']+1e-8)
        
        self.infos = [copy.deepcopy(info)]
        
        # Bookkeeping: setting values for next call:
        self.round_idx = (self.round_idx+1)%2

        if self.round_idx==0:
            self.stimulus_idx = (self.stimulus_idx+1)% self.dataloader_index2mode_length[self.dataloader_index] #len(data_loader)

            if self.stimulus_idx==0:
                self.dataloader_index = (self.dataloader_index+1)%len(self.dataloader_index2mode)
                self.mode = self.dataloader_index2mode[self.dataloader_index]
            
                if self.dataloader_index==0:
                    self.episode_ended = True
        
        return self.observations, self.infos

    def reset(self, **kwargs):
        self.mode = "train"
        self.done = False 

        # Step count since episode start
        self.step_count = 0

        # Which Dataloader ?
        self.dataloader_index = 0 
        #self.dataloader_index2mode = list(self.datasets.keys())
        self.dataloader_index2mode = ['train' for _ in range(self.nbr_shots)]
        #self.dataloader_index2mode += ['test']
        
        self.racc = [{'nbr_successes':0, 'nbr_games':0} for _ in self.dataloader_index2mode]

        self.previous_game_result = np.zeros((1,2))
        self.previous_game_was_successful = False
        
        # Return first observation
        obs, infos = self._gen_obs_info(reset=True)

        return obs, infos

    def step(self, action):
        self.step_count += 1

        self.action = action[0]

        reward = self._gen_reward()
        next_obs, next_infos = self._gen_obs_info()
        
        return next_obs, [reward], self.done, next_infos


    def _gen_reward(self):
        """
        Must imperatively be called before _gen_obs_info
        because it relies on the not-yet-updated value of round_idx.
        """
        reward = 0.0

        if self.round_idx==0:
            # then we have just received the listener's decision:
            if self.action == self.sample["speaker_exp_latents"][..., self.queried_attr_idx].item():
                self.previous_game_was_successful = True 
            else:
                self.previous_game_was_successful = False 

            if self.previous_game_was_successful:
                reward = 0.0
            else:
                if self.mode == "test":
                    reward = -2.0
                else:
                    reward = -1.0

            # accuracy bookkeeping:
            self.racc[self.dataloader_index]['nbr_games'] += 1
            self.racc[self.dataloader_index]['nbr_successes'] += int(self.previous_game_was_successful)
        
        if self.round_idx==0:
            self.previous_game_result = np.zeros((1,2))
            if self.previous_game_was_successful:
                self.previous_game_result[0,1] = 1
            else:
                self.previous_game_result[0,0] = 1

        return reward


def generate_recall_test_env(**kwargs):
    if kwargs.get('representation_type', "scr") == "ohe":
        kwargs["min_nbr_values_per_latent"] = kwargs["max_nbr_values_per_latent"]
 
    rg_config = kwargs.get('rg_config', None)
    if rg_config is None:
        rg_config = {
            "observability":            "partial",
            "max_sentence_length":      1,
            "nbr_communication_rounds": 1,
            "nbr_distractors":          {"train":1, "test":1},
            "distractor_sampling":      'uniform',
            # Default: use 'uniform' or "similarity-0.5"
            # otherwise the emerging language 
            # will have very high ambiguity...
            # Speakers find the strategy of uttering
            # a word that is relevant to the class/label
            # of the target, seemingly.  

            "descriptive":              False,
            "descriptive_target_ratio": 0.5,

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

    env = SymbolicBehaviourBenchmark_RecallTestEnv(**kwargs)

    return env 

def generate_recall_test_env_2shots(**kwargs):
    kwargs['nbr_shots'] = 2
    return generate_recall_test_env(**kwargs) 
