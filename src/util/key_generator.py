import random
def generate_sequence(n_samples, type, seed=47):
    if type == "sequential":
        return list(range(1, n_samples + 1))
    elif type == "constant":
        return [1] * n_samples
    elif type == "sampled":
        if seed is None:
            raise ValueError("A seed must be provided for sampled sequence generation.")
        rng = random.Random(seed)
        return rng.sample(range(1, 64001), n_samples)
    else:
        raise NotImplementedError(f"Key generation type {type} not implemented.")

def generate_keys(n_samples, id_type="sequential", kp_type="constant"):
    ids = generate_sequence(n_samples, id_type)
    k_ps = generate_sequence(n_samples, kp_type)
    return {"ids" : ids,
            "k_ps" : k_ps
            }