import fedjax
import jax
import jax.numpy as jnp
from typing import Any, Callable, Mapping, Sequence, Tuple

# FLIPS-specific utilities
# from utils.pruning import selective_layer_pruning
from fedjax import tree_util

ClientId = bytes
Grads = fedjax.Params

import jax
import jax.numpy as jnp

def selective_layer_pruning(params, importance_scores, pruning_thresholds):
    """
    A no-op pruning function that returns the original parameters unchanged.
    
    Args:
        params: The model parameters to consider for pruning.
        importance_scores: Dictionary of importance scores for each layer.
        pruning_thresholds: Dictionary of pruning thresholds for each layer.

    Returns:
        The original parameters, without any modifications.
    """
    return params

def calculate_pruning_thresholds(importance_scores, prune_quantile=0.5):
    """
    Calculate pruning thresholds based on importance scores.
    
    Args:
        importance_scores: Dictionary of importance scores for each layer.
        prune_quantile: Quantile for threshold calculation.
        
    Returns:
        A dictionary with thresholds for each layer.
    """
    layer_values = list(importance_scores.values())
    arr = jnp.array(layer_values)
    threshold_val = jnp.quantile(arr, prune_quantile)
    thresholds = {layer: threshold_val for layer in importance_scores}
    return thresholds


# import jax
# import jax.numpy as jnp

# def selective_layer_pruning(params, importance_scores, pruning_thresholds):
#     """Prune model layers selectively based on importance scores."""
#     pruned_params = {}
#     for layer, weights in params.items():
#         threshold = pruning_thresholds.get(layer, 0.0)
#         layer_importance = importance_scores.get(layer, 0.0)
#         if layer_importance < threshold:
#             pruned_params[layer] = jax.tree_util.tree_map(lambda w: 0.0, weights)
#         else:
#             pruned_params[layer] = weights
#     return pruned_params

# def calculate_pruning_thresholds(importance_scores, prune_quantile=0.5):
#     """Calculate pruning thresholds based on importance scores."""
#     layer_values = list(importance_scores.values())
#     arr = jnp.array(layer_values)
#     threshold_val = jnp.quantile(arr, prune_quantile)
#     thresholds = {layer: threshold_val for layer in importance_scores}
#     return thresholds



def compute_layer_importances(params):
    """Compute SHAP-like importance scores for each layer.

    This function handles both arrays and nested dictionaries of arrays.
    It recursively traverses through the structure, applying jnp.abs()
    and jnp.mean() only to the arrays, and sums their importances if there
    are nested levels.

    Args:
        params: A dictionary representing model parameters. Each entry can be
                either a jnp.array or another dictionary of arrays.

    Returns:
        A dictionary mapping layer names to their computed importance score (float).
    """
    def compute_layer_importance(layer_weights):
        if isinstance(layer_weights, dict):
            # If it's a dictionary, recursively compute the importance for each value
            sub_importances = [compute_layer_importance(sub_layer) for sub_layer in layer_weights.values()]
            return sum(sub_importances) / len(sub_importances) if sub_importances else 0.0
        else:
            # Here layer_weights should be a jnp.array
            return float(jnp.mean(jnp.abs(layer_weights)))

    importance_scores = {}
    for layer_name, layer_weights in params.items():
        importance_scores[layer_name] = compute_layer_importance(layer_weights)

    return importance_scores

@fedjax.dataclass
class FlipsServerState:
    """State of server passed between rounds for FLIPS."""
    params: fedjax.Params
    opt_state: fedjax.OptState

def compute_dynamic_client_weights(client_diagnostics, alpha=0.5, beta=0.5):
    weights = {}
    for client_id, diagnostics in client_diagnostics.items():
        grad_norm = diagnostics.get("delta_l2_norm", 0.0)
        metrics = diagnostics.get("metrics", [0.0])
        loss_reduction = metrics[0]  # Assuming first metric is loss improvement
        combined_score = alpha * grad_norm + beta * loss_reduction
        weights[client_id] = max(combined_score, 0.0)
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        raise ValueError("All computed weights are zero. Check client diagnostics.")
    return weights

def multifactor_client_selection_with_dynamic_weights(client_diagnostics):
    dynamic_weights = compute_dynamic_client_weights(client_diagnostics)
    return sorted(dynamic_weights.items(), key=lambda x: x[1], reverse=True)

def importance_weighted_aggregation(client_updates, importance_scores):
    total_weight = sum(importance_scores.values())
    if total_weight == 0:
        raise ValueError("Total importance weight is zero. Check importance_scores.")

    aggregated_update = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x), next(iter(client_updates.values()))
    )

    for client_id, update in client_updates.items():
        client_weight = importance_scores.get(client_id, 0.0) / total_weight
        aggregated_update = jax.tree_util.tree_map(
            lambda a, b: a + client_weight * b, aggregated_update, update
        )

    return aggregated_update


def flips_algorithm(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
) -> fedjax.FederatedAlgorithm:
    """FLIPS Algorithm."""

    def init(params: fedjax.Params) -> FlipsServerState:
        opt_state = server_optimizer.init(params)
        return FlipsServerState(params, opt_state)

    def compute_client_metrics(params, dataset):
        # Replace with actual metrics if needed.
        return [1.0, 2.0]

    def client_update_fn(server_params, client_dataset, client_rng, pruning_thresholds):
        params = server_params
        opt_state = client_optimizer.init(params)
        for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
            client_rng, use_rng = jax.random.split(client_rng)
            grads = grad_fn(params, batch, use_rng)
            opt_state, params = client_optimizer.apply(grads, opt_state, params)

        importance_scores = compute_layer_importances(params)
        pruned_params = selective_layer_pruning(params, importance_scores, pruning_thresholds)
        delta_params = jax.tree_util.tree_map(lambda a, b: a - b, server_params, pruned_params)
        diagnostics = {
            "delta_l2_norm": tree_util.tree_l2_norm(delta_params),
            "importance": importance_scores,
            "metrics": compute_client_metrics(pruned_params, client_dataset),
        }
        return delta_params, diagnostics

    def apply(server_state: FlipsServerState,
              clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]],
              pruning_thresholds: dict) -> Tuple[FlipsServerState, Mapping[ClientId, Any]]:
        client_updates = {}
        client_diagnostics = {}

        # Compute client updates
        for client_id, client_dataset, client_rng in clients:
            client_update, diagnostics = client_update_fn(
                server_state.params, client_dataset, client_rng, pruning_thresholds
            )
            client_updates[client_id] = client_update
            client_diagnostics[client_id] = diagnostics

        # Dynamic client selection based on diagnostics
        selected_clients = multifactor_client_selection_with_dynamic_weights(client_diagnostics)
        selected_client_ids = [cid for cid, _ in selected_clients]
        dynamic_weights = {cid: w for cid, w in selected_clients}

        importance_scores = {c: d["importance"] for c, d in client_diagnostics.items()}

        # Combine dynamic weights and importance scores
        flat_importance_scores = {}
        for cid in selected_client_ids:
            imp_sum = sum(float(v) for v in importance_scores[cid].values())
            flat_importance_scores[cid] = dynamic_weights[cid] * imp_sum

        selected_updates = {cid: client_updates[cid] for cid in selected_client_ids}
        aggregated_update = importance_weighted_aggregation(selected_updates, flat_importance_scores)

        # Server update
        opt_state, params = server_optimizer.apply(
            aggregated_update, server_state.opt_state, server_state.params
        )
        new_state = FlipsServerState(params, opt_state)
        return new_state, client_diagnostics

    return fedjax.FederatedAlgorithm(init, apply)

