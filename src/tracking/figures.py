import torch
import matplotlib.pyplot as plt


def plot_batch_loss(values):
    if not isinstance(values, list):
        values = [values]
    plt.cla()
    plt.close()
    figure = plt.figure(figsize=(20, 10))
    values = torch.tensor(values)
    plt.plot(values)
    plt.title("Batch Loss")
    plt.xlabel("batch")
    plt.ylabel("loss")
    return figure
