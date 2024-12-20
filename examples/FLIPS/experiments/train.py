import jax
import jax.numpy as jnp
import fedjax  # type: ignore
import time
import csv
import os
import matplotlib.pyplot as plt
from algorithms.factory import get_algorithm, ALGORITHMS
from utils.pruning import calculate_pruning_thresholds
from utils.plotting import plot_metrics
from utils.shap_utils import compute_layer_importances

# Supported datasets and how to load them, along with corresponding model functions.
# Note: For CIFAR-100, we do not have a built-in model in this snippet. We reuse the EMNIST conv model
# as a placeholder. In practice, you may want to define a proper CNN for CIFAR-100.
def load_emnist():
    train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)
    model = fedjax.models.emnist.create_conv_model(only_digits=False)
    return train_fd, test_fd, model

def load_cifar100():
    train_fd, test_fd = fedjax.datasets.cifar100.load_data()
    # Using EMNIST model as a generic CNN model placeholder:
    model = fedjax.models.cifar100.create_cnn_model()
    return train_fd, test_fd, model

def load_shakespeare():
    # load_data returns (train, held_out, test)
    train_fd, test_fd = fedjax.datasets.shakespeare.load_data()
    # We'll treat test_fd as the "test" set.
    model = fedjax.models.shakespeare.create_lstm_model()
    return train_fd, test_fd, model

def load_stackoverflow():
    # load_data returns (train, held_out, test)
    train_fd, held_out_fd, test_fd = fedjax.datasets.stackoverflow.load_data()
    model = fedjax.models.stackoverflow.create_lstm_model()
    return train_fd, test_fd, model

DATASETS = {
    "emnist": load_emnist,
    "cifar100": load_cifar100,
    "shakespeare": load_shakespeare,
    "stackoverflow": load_stackoverflow
}


def loss_fn(model):
    def loss(params, batch, rng):
        preds = model.apply_for_train(params, batch, rng)
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)
    return loss


def train_and_evaluate(algorithm_name, dataset_name):
    # If both algorithm and dataset are "all", run all combinations.
    if algorithm_name == "all" and dataset_name == "all":
        for ds in DATASETS.keys():
            for alg in ALGORITHMS:
                if alg == "all":
                    continue
                run_experiment(alg, ds)
        # Optionally, after all runs, you could create a global comparison plot.
        return
    elif algorithm_name == "all":
        # Run all algorithms on a single dataset
        if dataset_name not in DATASETS and dataset_name != "all":
            raise ValueError(f"Dataset {dataset_name} not supported.")
        if dataset_name == "all":
            # Run all datasets with all algorithms (already handled above)
            for ds in DATASETS.keys():
                for alg in ALGORITHMS:
                    if alg == "all":
                        continue
                    run_experiment(alg, ds)
        else:
            for alg in ALGORITHMS:
                if alg == "all":
                    continue
                run_experiment(alg, dataset_name)
    elif dataset_name == "all":
        # Run a single algorithm on all datasets
        if algorithm_name not in ALGORITHMS and algorithm_name != "all":
            raise ValueError(f"Algorithm {algorithm_name} not supported.")
        for ds in DATASETS.keys():
            run_experiment(algorithm_name, ds)
    else:
        # Run a single algorithm on a single dataset
        run_experiment(algorithm_name, dataset_name)


def run_experiment(algorithm_name, dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Load dataset and model
    train_fd, test_fd, model = DATASETS[dataset_name]()

    grad_fn = jax.jit(jax.grad(loss_fn(model)))
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4)
    )
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    # Run training rounds
    rounds, accuracies = run_training_rounds(
        algorithm_name,
        model,
        grad_fn,
        client_optimizer,
        server_optimizer,
        client_batch_hparams,
        train_fd,
        test_fd,
        dataset_name
    )

    # If "all" algorithms and datasets are run, a global comparison plot is done outside of this function.
    # Here, each run_experiment call already saves its own CSV and plot.


def run_training_rounds(
    algorithm_name,
    model,
    grad_fn,
    client_optimizer,
    server_optimizer,
    client_batch_hparams,
    train_fd,
    test_fd,
    dataset_name
):
    algorithm = get_algorithm(
        algorithm_name,
        grad_fn,
        client_optimizer,
        server_optimizer,
        client_batch_hparams,
    )

    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=train_fd, num_clients=10, seed=0
    )
    test_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=test_fd, num_clients=10, seed=0
    )

    # Lists to store metrics for logging
    train_accuracies, test_accuracies = [], []
    train_losses, test_losses = [], []
    rounds = []
    num_clients_per_round = []
    round_times = []
    
    overall_start_time = time.time()

    # Prepare CSV logging
    csv_filename = f"{algorithm_name}_{dataset_name}_results.csv"
    # If file exists, remove it so we don't append to old data
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = [
            'round',
            'train_accuracy',
            'test_accuracy',
            'train_loss',
            'test_loss',
            'user_participation',
            'round_time'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for round_num in range(1, 10):
            clients = train_client_sampler.sample()
            client_ids = [cid for cid, _, _ in clients]
            test_client_ids = [cid for cid, _, _ in test_client_sampler.sample()]
            num_participants = len(client_ids)

            # Measure the time taken by the algorithm per round
            apply_start_time = time.time()
            if algorithm_name == "flips":
                # Compute pruning thresholds for FLIPS
                importance_scores = compute_layer_importances(server_state.params)
                pruning_thresholds = calculate_pruning_thresholds(importance_scores, prune_quantile=0.5)

                server_state, client_diagnostics = algorithm.apply(server_state, clients, pruning_thresholds)
            else:
                server_state, client_diagnostics = algorithm.apply(server_state, clients)
            apply_end_time = time.time()
            round_time = apply_end_time - apply_start_time

            # Evaluate
            train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
            test_eval_datasets = [cds for _, cds in test_fd.get_clients(test_client_ids)]
            train_eval_batches = fedjax.padded_batch_client_datasets(
                train_eval_datasets, batch_size=256
            )
            test_eval_batches = fedjax.padded_batch_client_datasets(
                test_eval_datasets, batch_size=256
            )

            train_metrics = fedjax.evaluate_model(
                model, server_state.params, train_eval_batches
            )
            test_metrics = fedjax.evaluate_model(
                model, server_state.params, test_eval_batches
            )

            if dataset_name == "shakespeare":
                train_metrics = {
                    'accuracy': train_metrics['accuracy_in_vocab'],  # Renaming 'accuracy_in_vocab' to 'accuracy'
                    'loss': train_metrics['sequence_loss'],         # Renaming 'sequence_loss' to 'loss'
                    'accuracy_in_vocab': train_metrics['accuracy_in_vocab'],
                    'accuracy_no_eos': train_metrics['accuracy_no_eos'],
                    'num_tokens': train_metrics['num_tokens'],
                    'sequence_length': train_metrics['sequence_length'],
                    'sequence_loss': train_metrics['sequence_loss'],
                    'token_loss': train_metrics['token_loss'],
                    'token_oov_rate': train_metrics['token_oov_rate']
                }
                test_metrics = {
                    'accuracy': test_metrics['accuracy_in_vocab'],  # Renaming 'accuracy_in_vocab' to 'accuracy'
                    'loss': test_metrics['sequence_loss'],          # Renaming 'sequence_loss' to 'loss'
                    'accuracy_in_vocab': test_metrics['accuracy_in_vocab'],
                    'accuracy_no_eos': test_metrics['accuracy_no_eos'],
                    'num_tokens': test_metrics['num_tokens'],
                    'sequence_length': test_metrics['sequence_length'],
                    'sequence_loss': test_metrics['sequence_loss'],
                    'token_loss': test_metrics['token_loss'],
                    'token_oov_rate': test_metrics['token_oov_rate']
                }


            train_accuracies.append(train_metrics["accuracy"])
            test_accuracies.append(test_metrics["accuracy"])
            train_losses.append(train_metrics["loss"])
            test_losses.append(test_metrics["loss"])
            rounds.append(round_num)
            num_clients_per_round.append(num_participants)
            round_times.append(round_time)

            print(f"[{dataset_name} | {algorithm_name} | round {round_num}] train_metrics={train_metrics}")
            print(f"[{dataset_name} | {algorithm_name} | round {round_num}] test_metrics={test_metrics}")

            # Write round results to CSV
            writer.writerow({
                'round': round_num,
                'train_accuracy': train_metrics["accuracy"],
                'test_accuracy': test_metrics["accuracy"],
                'train_loss': train_metrics["loss"],
                'test_loss': test_metrics["loss"],
                'user_participation': num_participants,
                'round_time': round_time
            })

    # Save plots for this algorithm-dataset combination
    plot_metrics(rounds, train_accuracies, test_accuracies, train_losses, test_losses, f"{algorithm_name}_{dataset_name}")

    overall_time = time.time() - overall_start_time
    print(f"Total training time for {algorithm_name} on {dataset_name}: {overall_time} seconds")

    # Append overall training time to the CSV file as a summary row (optional)
    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = [
            'round',
            'train_accuracy',
            'test_accuracy',
            'train_loss',
            'test_loss',
            'user_participation',
            'round_time'
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({
            'round': 'total',
            'train_accuracy': '',
            'test_accuracy': '',
            'train_loss': '',
            'test_loss': '',
            'user_participation': '',
            'round_time': overall_time
        })

    return rounds, test_accuracies
