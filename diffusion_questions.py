import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from diffusion_models import *

SAMPLE_DIM = 2
TRAINING_SIZE = 3000
EPOCHS = 3000

def split_points_for_plot(points):
    x = [points[i][0][0] for i in range(len(points))]
    y = [points[i][0][1] for i in range(len(points))]
    return x, y



if __name__ == '__main__':

    # Unconditioned model
    T = 1000

    # Q1
    point = torch.tensor([0.,0.])
    progress = [point.tolist()]
    t_steps = torch.arange(0, 1, 1/T)
    for t in t_steps:
        progress.append(forward_process(point, t, batch=1)[0].tolist())
    point = forward_process(point, torch.tensor(1), 1)
    progress.append(point[0].tolist())
    plt.scatter(*zip(*progress[1:]), c=t_steps.tolist()+[1])
    plt.colorbar().set_label("t")
    plt.title("forward process on point with color indicating t from (0,0)")
    plt.show()


    # Q2

    training_data = PointData(-1, 1, SAMPLE_DIM, TRAINING_SIZE)
    model = Denoiser()
    running_loss = list(range(2999))
    model.train()
    running_loss = train_denoiser(model, training_data, scheduler)
    model.eval()
    model.eval()
    model.requires_grad_(False)
    plt.plot(list(range(EPOCHS)), running_loss,
             label="running loss")
    plt.title("Loss Function Over Training Unconditional")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Q3
    sample_size = 1000
    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    for i, ax in enumerate(fig.axes):
        torch.manual_seed(i)
        points = ddim_sampling(model, -1 / T, scheduler,
                                            sample_size=sample_size)
        ax.scatter(points.detach().numpy()[:,0], points.detach().numpy()[:,
                                                 1],alpha=0.5)
        ax.set_title(f"seed - {i}")
    plt.suptitle('9 different 1000 point samples from different seeds')
    plt.show()


    # Q4
    T_vals = [50, 100, 200, 500, 1000, 2000, 5000]
    points = []
    torch.manual_seed(0)
    z = torch.randn(1, SAMPLE_DIM)
    for T in T_vals:
        point = ddim_sampling(model, -1 / T, scheduler, z=z.clone()).tolist()
        plt.scatter(point[0][0], point[0][1], alpha=0.5, label=str(T))
    plt.legend()
    plt.title("sampled points from different T's")
    plt.show()

    # Q5
    T = 1000
    modded_scheduler = lambda t: t
    z = torch.randn(1, SAMPLE_DIM)
    # fig, axs = plt.subplots(1, 2, figsize=(9,9))
    # axs[0,1] =
    mod_sample, movement = ddim_sampling(model, -1 / T, modded_scheduler,
                                         track=True, z=z.clone())
    plt.scatter(*split_points_for_plot(movement), label="modified - t", s=15,
                alpha=0.5)
    sample, movement = ddim_sampling(model, -1 / T, scheduler, track=True,
                                     z=z.clone())
    plt.scatter(*split_points_for_plot(movement), label="original - "
                                                        "exp(5("
                                                        "t-1)", s=15,
                alpha=0.5)
    plt.title("comparing different schedulers")
    plt.legend()
    plt.show()
    fig, axes = plt.subplots(1, 2)
        # torch.manual_seed(10)
    points = ddim_sampling(model, -1 / T, scheduler,
                                        sample_size=1000)
    axes[0].scatter(points.detach().numpy()[:,0], points.detach().numpy(

    )[:,
                                             1],alpha=0.5)
    axes[0].set_title("original sampler - exp(5(t-1)")
    points = ddim_sampling(model, -1 / T, modded_scheduler,
                           sample_size=1000)
    axes[1].scatter(points.detach().numpy()[:, 0], points.detach().numpy(

    )[:,
                                                      1], alpha=0.5)
    axes[1].set_title("modified sampler - t")

    fig.suptitle("sampling from different schedulers")
    fig.show()





    # Q6
    z = torch.randn(1, SAMPLE_DIM)
    samples = []
    for _ in range(10):
        samples.append(ddim_sampling(model, -1/T, scheduler,z=z.clone(), noisy=False).tolist())
    plt.scatter(*split_points_for_plot(samples),c=list(range(10)))
    plt.title("reverse sampling the same noise 10 times (they are the same)")
    plt.show() # they are the same
    for _ in range(4):
        point, traj = ddim_sampling(model, -1 / T, scheduler, z=z.clone(),
                            track=True, noisy=True)
        plt.scatter(*split_points_for_plot(traj))
    plt.title("reverse sampling the same noise 4 times with added noise")
    plt.show()



    # Questions:
    # Conditional model
    # Q1
    num_classes = 6
    model = ConditionalDenoiser(num_classes)
    conditional_data = PointData(-1, 1, SAMPLE_DIM, TRAINING_SIZE,
                                 class_func=get_class)

    x = [conditional_data[i][0][0].item() for i in range(len(
        conditional_data))]
    y = [conditional_data[i][0][1].item() for i in range(len(
        conditional_data))]
    cmap = plt.get_cmap('RdBu', 6)
    plt.scatter(x, y, c=conditional_data.class_data.tolist(),
                cmap=cmap)
    plt.colorbar()
    plt.title("Conditional data colored by class")
    plt.show()


    running_loss = list(range(TRAINING_SIZE))
    model.train()
    running_loss = train_denoiser(model, conditional_data, scheduler,
                                  conditioned=True)
    model.eval()
    plt.plot(list(range(EPOCHS)), running_loss)
    plt.title("conditioned model loss")
    plt.show()

    # Q2 in pdf

    # Q3
    classes = list(range(num_classes))
    for c in classes:
        point, trajectory = ddim_sampling(model, -1 / T, scheduler,
                                          condition=torch.tensor([c]),
                                          track=True)

        plt.scatter(*split_points_for_plot(trajectory),s=15,alpha=0.6,
        label=str(c))
    plt.axvline(x=0)
    plt.axvline(x=0.5)
    plt.axhline(y=0)
    plt.legend()
    plt.title("Trajectory of points sampled from different classes")
    plt.show()

    # Q4
    classes = torch.empty(0, dtype=torch.int8)
    for i in range(num_classes):
        size = 166
        size += 4 if i==5 else 0
        samples = ddim_sampling(model, -1/T,scheduler, size,
                                condition=torch.tensor([i]))
        x = [samples[i][0].item() for i in range(len(samples))]
        y = [samples[i][1].item() for i in range(len(samples))]
        plt.scatter(x, y, label=str(i),alpha=0.5)

    plt.legend()
    plt.title("1000 samples colored by class")
    plt.show()

    # Q5 - in pdf

    # Q6

    points = [[-1.361, -1.4922],[-0.7672, 0.7763],[-0.7472, 0.7963],
            [0.7768, -0.0869],[0.3016, 0.4217],[-0.0546, -0.9501]]
    classes = [torch.tensor(get_class(x)) for x in points]
    classes[2] = torch.tensor(3)
    x = [points[i][0] for i in range(len(points))]
    y = [points[i][1] for i in range(len(points))]
    plt.scatter(x, y, c=classes, s=25)
    log_probs = []
    for i in range(6):
        log_probs.append(round(prob_estimastion(model, torch.tensor(points[
                                                                        i]),
                                            c=classes[i]).item(),3))
        if i == 2: plt.annotate(log_probs[i], (x[i]+0.1,y[i]-0.1),
                                fontsize=16)
        else:
            plt.annotate(log_probs[i], (x[i],y[i]+0.1), fontsize=16)
    x = [conditional_data[i][0][0].item() for i in range(len(
        conditional_data))]
    y = [conditional_data[i][0][1].item() for i in range(len(conditional_data))]
    plt.scatter(x, y, c=conditional_data.class_data.detach(), s=5, alpha=0.2)
    print(log_probs)
    plt.colorbar()
    plt.title("plotting 5 points and displaying their probability colored by class")
    plt.show()


