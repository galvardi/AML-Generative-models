import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tqdm import trange


SHRINK_FACTOR = 0.3
SAMPLE_DIM = 2
TRAINING_SIZE = 3000
EPOCHS = 3000



class PointData(Dataset):
    def __init__(self, low_bound, high_bound, sample_dim, dataset_size, class_func=None):
        self.min = low_bound
        self.max = high_bound
        self.dim = sample_dim
        self.size = dataset_size
        self.class_func = class_func
        self.data = (low_bound - high_bound) * torch.rand(dataset_size,
                                                      sample_dim,
                                                requires_grad=True) + high_bound

        if class_func is not None:

            self.class_data = torch.from_numpy(np.apply_along_axis(class_func, 1,
                                                self.data.detach().numpy()))


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.class_func is not None:
            return self.data[index], self.class_data[index]
        return self.data[index]

def get_class(x):
    left = True if x[0] < 0 else False
    mid = not left and x[0] < 0.5
    top = True if x[1] > 0 else False
    if left:
        if top: return 0
        return 1
    if mid:
        if top: return 2
        return 3
    if top: return 4
    return 5
 # if left:
 #        if top: return float(0)
 #        return float(1)
 #    if mid:
 #        if top: return float(2)
 #        return float(3)
 #    if top: return float(4)
 #    return float(5)


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.fc1 = nn.Linear(SAMPLE_DIM + 1, 128)  # condition on t
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, SAMPLE_DIM)
        self.activation = nn.functional.leaky_relu

    def forward(self, x, t):
        x_t = torch.hstack([x, t])
        x_t = self.activation(self.fc1(x_t))
        x_t = self.activation(self.fc2(x_t))
        return self.activation(self.fc3(x_t))


class ConditionalDenoiser(nn.Module):
    def __init__(self, classes):
        super(ConditionalDenoiser, self).__init__()
        self.embedding = nn.Embedding(classes, 10)
        self.fc_in = nn.Linear(13, 128)  # merge condition embedding
        self.fc1 = nn.Linear(128, 128)  # condition on t
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.activation = nn.functional.leaky_relu

    def forward(self, x, t, c):
        embd = self.embedding(c)
        x_t = torch.hstack([x, t])
        x_t = torch.hstack([x_t, embd])
        x_t = self.activation(self.fc_in(x_t))
        x_t = self.activation(self.fc1(x_t))
        x_t = self.activation(self.fc2(x_t))
        return self.activation(self.fc3(x_t))



def train_denoiser(model, training_set, scheduler, conditioned=False):
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=
                                                  TRAINING_SIZE,
                                                  shuffle=True)
    loss_fn = nn.MSELoss()
    running_loss = []
    model.train()
    for epoch in trange(EPOCHS):
        for data in training_loader:
            points = data[0]
            optimizer.zero_grad()
            epsilons = torch.randn([TRAINING_SIZE, SAMPLE_DIM])
            t_values = torch.rand([TRAINING_SIZE, 1])
            # sigmas = torch.exp(5 * (t_values - 1))
            sigmas = scheduler(t_values)
            x_t = points + torch.mul(sigmas, epsilons)  # adding noise
            if conditioned:
                pred_epsilon = model(x_t, t_values, data[1].long())
            else:
                pred_epsilon = model(x_t, t_values)
            loss = loss_fn(pred_epsilon, epsilons)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss.append(loss.item())

    return running_loss


def scheduler(t):
    return torch.exp(5 * (t - 1)).requires_grad_(True)


def forward_process(x, t ,batch=1, epsilons=None): #todo check
    if epsilons is None: epsilons = torch.randn(batch, SAMPLE_DIM)
    sigmas = torch.exp(5 * (t - 1)).requires_grad_(True)
    return x + sigmas*epsilons

def prob_estimastion(denoiser, x, c=None):  # todo do
    T = 100
    num_combinations = 1000
    dt = -1 / T
    noise = torch.randn(num_combinations, SAMPLE_DIM)
    t_vals = torch.rand(num_combinations, 1)
    x = x.reshape(1, 2)
    x = x.expand(num_combinations, 2)
    x_t = x + (scheduler(t_vals)*noise)
    c = c.expand(num_combinations, )
    x_0 = x_t - scheduler(t_vals) * denoiser(x_t, t_vals, c)
    dif = torch.norm(x - x_0, dim=1, keepdim=True).pow(2)
    snr_vals = 1 / (scheduler(t_vals).pow(2))
    snr_dt_vals = 1 / (scheduler(t_vals - dt).pow(2))
    snr_vals = snr_dt_vals - snr_vals
    return torch.mul(snr_vals, dif).mean() * (T / 2)


def ddim_sampling(denoiser, dt, sigma, sample_size=1, track=False, z=None,
                  noisy=False, condition=None):
    if z is None : z = torch.randn(sample_size, SAMPLE_DIM)
    if track: movement = [z.detach().tolist()]
    for t in torch.arange(1, 0, dt):
        t = torch.tensor([t], requires_grad=True).view(1,1)
        t.retain_grad()
        # getting gradient of schedular
        sigma_grad = sigma(t)
        sigma_grad.backward()
        t_grad = t.grad.item()
        if condition is None:
            pred_noise = denoiser(z, t.expand(sample_size, 1))
        else:
            pred_noise = denoiser(z, t.expand(sample_size, 1),
                                        condition.expand(sample_size))
        z_hat = z - sigma(t) * pred_noise
        if noisy:
            z_hat += sigma(t) * SHRINK_FACTOR * torch.randn(
                z_hat.shape[0], SAMPLE_DIM)
        score = (z_hat - z) / (sigma(t) ** 2)
        dz = -(t_grad * sigma(t)) * score * dt
        if track:
            movement.append(z.detach().tolist())
        z += dz
    if track: return z, movement
    return z