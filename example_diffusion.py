import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from diffusion_models import *

SAMPLE_DIM = 2
TRAINING_SIZE = 3000
EPOCHS = 3000

if __name__ == '__main__':
    # unconditional
    training_data = PointData(-1, 1, SAMPLE_DIM, TRAINING_SIZE)
    model = Denoiser()
    running_loss = list(range(2999))
    model.train()
    running_loss = train_denoiser(model, training_data, scheduler)
    model.eval()
    model.requires_grad_(False)
    plt.plot(list(range(EPOCHS)), running_loss,
             label="running loss")
    plt.title("Loss Function Over Training Unconditional")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    T = 1000
    sample_size = 1000
    points = ddim_sampling(model, -1 / T, scheduler,
                          sample_size=sample_size)
    plt.scatter(points.detach().numpy()[:, 0], points.detach().numpy()[:,
                                            1], alpha=0.5)
    plt.title("sampling unconditional")
    plt.show()

    # conditional

    num_classes = 6
    model = ConditionalDenoiser(num_classes)
    conditional_data = PointData(-1, 1, SAMPLE_DIM, TRAINING_SIZE,
                                 class_func=get_class)
    running_loss = list(range(TRAINING_SIZE))
    model.train()
    running_loss = train_denoiser(model, conditional_data, scheduler,
                                  conditioned=True)
    model.eval()
    plt.plot(list(range(EPOCHS)), running_loss)
    plt.title("conditioned model loss")
    plt.show()

    classes = torch.empty(0, dtype=torch.int8)
    for i in range(num_classes):
        size = 166
        size += 4 if i == 5 else 0
        samples = ddim_sampling(model, -1 / T, scheduler, size,
                                condition=torch.tensor([i]))
        x = [samples[i][0].item() for i in range(len(samples))]
        y = [samples[i][1].item() for i in range(len(samples))]
        plt.scatter(x, y, label=str(i), alpha=0.5)

    plt.legend()
    plt.title("conditioned sampling")
    plt.show()
