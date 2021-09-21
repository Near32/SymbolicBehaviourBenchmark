import gym
import symbolic_behaviour_benchmark
import numpy as np 
import random

def test_env(
    nbr_latents=2, 
    min_nbr_values_per_latent=4, 
    max_nbr_values_per_latent=5,
    nbr_object_centric_samples=1,
    ):
    
    env = gym.make(
        "SymbolicBehaviourBenchmark-RecallTestEnv-v0",
        representation_type="scr",
        nbr_latents=nbr_latents,
        min_nbr_values_per_latent=min_nbr_values_per_latent,
        max_nbr_values_per_latent=max_nbr_values_per_latent,
        nbr_object_centric_samples=nbr_object_centric_samples,
    )
    
    obs, info = env.reset()
    
    import ipdb; ipdb.set_trace()
    
    # NO-OP:
    action = [max_nbr_values_per_latent]
    output = env.step(action=action)

    import ipdb; ipdb.set_trace()

    action = [0]
    foutput = env.step(action=action)

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

    test_env()
