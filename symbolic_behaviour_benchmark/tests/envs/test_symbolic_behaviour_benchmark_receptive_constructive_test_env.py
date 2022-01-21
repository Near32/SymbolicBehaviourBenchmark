import gym
import symbolic_behaviour_benchmark
import numpy as np 
import random

def test_env(
    vocab_size=10,
    max_sentence_length=5,
    nbr_latents=2, 
    min_nbr_values_per_latent=4, 
    max_nbr_values_per_latent=5,
    nbr_object_centric_samples=1,
    nbr_distractors=3,
    allow_listener_query=False,
    use_communication_channel_permutations=True,
    ):
    
    """
    rg_config = {
        "observability":            "full",
        "max_sentence_length":      max_sentence_length,
        "vocab_size":               vocab_size,
        "nbr_communication_round":  1,
    
        "nbr_distractors":          {"train":nbr_latents, "test":nbr_distractors},
        "distractor_sampling":      'uniform',
        # Default: use 'uniform' or "similarity-0.5"
        # otherwise the emerging language 
        # will have very high ambiguity...
        # Speakers find the strategy of uttering
        # a word that is relevant to the class/label
        # of the target, seemingly.  

        "descriptive":              False,
        "descriptive_target_ratio": 0.5,

        "object_centric":           False,
        "nbr_stimulus":             1,

        "use_curriculum_nbr_distractors": False,
        "curriculum_distractors_window_size": 25, #100,
    }

    train_dataset = SymbolicContinuousStimulusDataset(
        train=True,
        transform=None,
        split_strategy='combinatorial2-40',
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=1,
        prototype=None,
    )
    """

    env = gym.make(
        "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0", 
        #rg_config=rg_config,
        #train_dataset=train_dataset,
        vocab_size=vocab_size,
        max_sentence_length=max_sentence_length,
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
        nbr_distractors=nbr_distractors,
        use_communication_channel_permutations=use_communication_channel_permutations,
        allow_listener_query=allow_listener_query,
    )
    
    obs, info = env.reset()
    
    import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones(5)*3}
    listener_action = {'decision':2, 'communication_channel': np.ones(5)*2}
    output = env.step(action=[speaker_action, listener_action])

    import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones(5)*3}
    listener_action = {'decision':0, 'communication_channel': np.ones(5)*2}
    foutput = env.step(action=[speaker_action, listener_action])

    import ipdb; ipdb.set_trace()

    env.close()


def test_env_sampling_strategy(
    vocab_size=10,
    max_sentence_length=5,
    nbr_latents=2, 
    min_nbr_values_per_latent=4, 
    max_nbr_values_per_latent=5,
    nbr_object_centric_samples=1,
    nbr_distractors=3,
    allow_listener_query=False,
    use_communication_channel_permutations=True,
    sampling_strategy='component-focused-3shots',
    ):
    
    """
    rg_config = {
        "observability":            "full",
        "max_sentence_length":      max_sentence_length,
        "vocab_size":               vocab_size,
        "nbr_communication_round":  1,
    
        "nbr_distractors":          {"train":nbr_latents, "test":nbr_distractors},
        "distractor_sampling":      'uniform',
        # Default: use 'uniform' or "similarity-0.5"
        # otherwise the emerging language 
        # will have very high ambiguity...
        # Speakers find the strategy of uttering
        # a word that is relevant to the class/label
        # of the target, seemingly.  

        "descriptive":              False,
        "descriptive_target_ratio": 0.5,

        "object_centric":           False,
        "nbr_stimulus":             1,

        "use_curriculum_nbr_distractors": False,
        "curriculum_distractors_window_size": 25, #100,
    }

    train_dataset = SymbolicContinuousStimulusDataset(
        train=True,
        transform=None,
        split_strategy='combinatorial2-40',
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=1,
        prototype=None,
    )
    """

    env = gym.make(
        "SymbolicBehaviourBenchmark-ReceptiveConstructiveTestEnv-v0", 
        #rg_config=rg_config,
        #train_dataset=train_dataset,
        vocab_size=vocab_size,
        max_sentence_length=max_sentence_length,
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
        nbr_distractors=nbr_distractors,
        use_communication_channel_permutations=use_communication_channel_permutations,
        allow_listener_query=allow_listener_query,
        sampling_strategy=sampling_strategy,
    )
    
    obs, info = env.reset()
    
    import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
    listener_action = {'decision':2, 'communication_channel': np.ones((1,5))*2}
    output = env.step(action=[speaker_action, listener_action])

    import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
    listener_action = {'decision':0, 'communication_channel': np.ones((1,5))*2}
    foutput = env.step(action=[speaker_action, listener_action])

    import ipdb; ipdb.set_trace()

    env.close()

if __name__ == "__main__":
    seed = 1 
    # Following: https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    if hasattr(torch.backends, "cudnn") and not(args.fast):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    """

    np.random.seed(seed)
    random.seed(seed)

    #test_env()
    test_env_sampling_strategy()
