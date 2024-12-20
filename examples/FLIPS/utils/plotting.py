import matplotlib.pyplot as plt

def plot_metrics(rounds, train_accuracies, test_accuracies, train_losses, test_losses, algorithm_name):
    plt.figure()
    plt.plot(rounds, train_accuracies, label="Train Accuracy")
    plt.plot(rounds, test_accuracies, label="Test Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Rounds - {algorithm_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{algorithm_name}_accuracy.png")

    plt.figure()
    plt.plot(rounds, train_losses, label="Train Loss")
    plt.plot(rounds, test_losses, label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss over Rounds - {algorithm_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{algorithm_name}_loss.png")

