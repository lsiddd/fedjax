from absl import app, flags  # type: ignore
from algorithms.factory import ALGORITHMS
from experiments.train import train_and_evaluate

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "algorithm", "flips", 
    "The algorithm to use: fednova, fedprox, fedavg, flips, or all."
)
flags.DEFINE_string(
    "dataset", "cifar100", 
    "The dataset to use: emnist, cifar100, shakespeare, stackoverflow, or all."
)

def main(_):
    algorithm = FLAGS.algorithm
    dataset = FLAGS.dataset

    # Run training and evaluation
    train_and_evaluate(algorithm, dataset)

if __name__ == "__main__":
    app.run(main)
