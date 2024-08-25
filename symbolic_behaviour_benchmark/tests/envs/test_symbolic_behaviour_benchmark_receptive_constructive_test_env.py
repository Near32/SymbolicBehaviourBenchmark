import gym
import symbolic_behaviour_benchmark
import numpy as np 
import random

from symbolic_behaviour_benchmark.utils.utils import STR2BT, BT2STR


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

def test_env_descr_feedback(
    vocab_size=10,
    max_sentence_length=5,
    descriptive=True,
    nbr_latents=2, 
    nbr_communication_rounds=1,
    min_nbr_values_per_latent=4, 
    max_nbr_values_per_latent=5,
    nbr_object_centric_samples=1,
    nbr_distractors=0,
    allow_listener_query=False,
    provide_listener_feedback=True,
    use_communication_channel_permutations=True,
    sampling_strategy='component-focused-1shot',
    render=False,
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
        descriptive=descriptive,
        nbr_communication_rounds=nbr_communication_rounds,
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
        nbr_distractors=nbr_distractors,
        use_communication_channel_permutations=use_communication_channel_permutations,
        allow_listener_query=allow_listener_query,
        provide_listener_feedback=provide_listener_feedback,
        sampling_strategy=sampling_strategy,
    )
    
    obs, info = env.reset()
    if render:
        env.render(mode='human') 
    #import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
    listener_action = {'decision':2, 'communication_channel': np.ones((1,5))*2}
    output = env.step(action=[speaker_action, listener_action])

    #import ipdb; ipdb.set_trace()
    
    ok = True
    while ok:
        if render:
            env.render(mode='human')
        speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
        listener_action = {'decision':0, 'communication_channel': np.ones((1,5))*2}
        foutput = env.step(action=[speaker_action, listener_action])
        done = foutput[2]
        ok = done==False
        
    env.close()

def test_env_llm_prompt(
    vocab_size=10,
    max_sentence_length=5,
    descriptive=True,
    nbr_latents=3, 
    nbr_communication_rounds=1,
    min_nbr_values_per_latent=9, 
    max_nbr_values_per_latent=10,
    nbr_object_centric_samples=1,
    nbr_distractors=0,
    allow_listener_query=False,
    provide_listener_feedback=True,
    use_communication_channel_permutations=True,
    sampling_strategy='component-focused-2shots',
    render=False,
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
        descriptive=descriptive,
        nbr_communication_rounds=nbr_communication_rounds,
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
        nbr_distractors=nbr_distractors,
        use_communication_channel_permutations=use_communication_channel_permutations,
        allow_listener_query=allow_listener_query,
        provide_listener_feedback=provide_listener_feedback,
        sampling_strategy=sampling_strategy,
        include_prompts=True,
    )
    
    obs, info = env.reset()
    
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
    import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
    listener_action = {'decision':1, 'communication_channel': np.ones((1,5))*2}
    output = env.step(action=[speaker_action, listener_action])

    info = output[-1]
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
    import ipdb; ipdb.set_trace()
    
    output = env.step(action=[speaker_action, listener_action])

    info = output[-1]
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
    import ipdb; ipdb.set_trace()

    output = env.step(action=[speaker_action, listener_action])

    info = output[-1]
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
    import ipdb; ipdb.set_trace()

    ok = True
    it = 0
    foutput = output
    while ok:
        it = foutput[-1][0]['stimulus_idx']
        it = it % vocab_size
        #if render:
        #    env.render(mode='human')
        speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*it}
        listener_action = {'decision':0, 'communication_channel': np.ones((1,5))*2}
        foutput = env.step(action=[speaker_action, listener_action])
        done = foutput[2]
        ok = done==False 
        print(foutput[-1][0]['prompt'].shape[-1],foutput[-1][1]['prompt'].shape[-1])
    
    print(BT2STR(foutput[-1][0]['prompt'])[0])
    import ipdb; ipdb.set_trace()
    print(BT2STR(foutput[-1][1]['prompt'])[0])
    import ipdb; ipdb.set_trace()
    
    obs, info = env.reset()
    
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
    #import ipdb; ipdb.set_trace()

    speaker_action = {'decision':0, 'communication_channel': np.ones((1,5))*3}
    listener_action = {'decision':2, 'communication_channel': np.ones((1,5))*2}
    output = env.step(action=[speaker_action, listener_action])

    info = output[-1]
    print(BT2STR(info[0]['prompt'])[0])
    print('\n'+'-'*10+'\n')
    print(BT2STR(info[1]['prompt'])[0])
    
    if render:
        env.render(mode='human') 
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
    #test_env_sampling_strategy()
    #test_env_descr_feedback()
    #test_env_descr_feedback(nbr_object_centric_samples=2, render=True)
    test_env_llm_prompt(nbr_object_centric_samples=2, render=False)
