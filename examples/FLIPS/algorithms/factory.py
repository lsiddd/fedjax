import fedjax
import jax
import jax.numpy as jnp
from algorithms import fed_avg, fed_prox, fednova, flips

ALGORITHMS = ["fedprox", "fednova", "fedavg", "flips"]

def get_algorithm(name, grad_fn, client_optimizer, server_optimizer, client_batch_hparams):
    if name == "fednova":
        return fednova.federated_nova(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams
        )
    elif name == "fedprox":
        return fed_prox.federated_proximal(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams, mu=0.1
        )
    elif name == "fedavg":
        return fed_avg.federated_averaging(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams
        )
    elif name == "flips":
        return flips.flips_algorithm(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams
        )
    else:
        raise ValueError(f"Unsupported algorithm: {name}")

