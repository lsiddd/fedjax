import jax.numpy as jnp

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
