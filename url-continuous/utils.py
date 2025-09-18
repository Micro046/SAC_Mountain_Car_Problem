import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, window: int = 100):
    """
    Plot both raw scores and a running average over the previous `window` episodes.

    Arguments:
    - x: array-like of x-values (e.g. timesteps or episode indices)
    - scores: list or array of raw episodic returns
    - figure_file: path to save the PNG file
    - window: integer, window size for computing the running average

    The resulting figure will show:
      1) Light-gray line for raw episodic returns
      2) Solid blue line for the running average of length == window
      3) A red dot marking the maximum of the running average
    """
    scores = np.array(scores)
    running_avg = np.zeros_like(scores, dtype=float)

    for i in range(len(scores)):
        low = max(0, i - window + 1)
        running_avg[i] = np.mean(scores[low : i + 1])

    plt.figure(figsize=(10, 5))
    # 1) Plot raw episodic rewards in light gray
    plt.plot(x, scores, color='lightgray', label='Raw Episode Reward', linewidth=1)

    # 2) Plot running average in solid blue
    plt.plot(x, running_avg, color='tab:blue', label=f'{window}-Episode Running Average', linewidth=2)

    # 3) Mark the maximum running average
    max_idx = np.argmax(running_avg)
    max_x = x[max_idx]
    max_y = running_avg[max_idx]
    plt.scatter([max_x], [max_y], color='red', s=50, zorder=5, label=f'Peak Avg ({max_y:.2f})')

    # Axis labels and title
    plt.xlabel('Episode' if len(x) == len(scores) else 'Timestep', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Learning Curve: Raw Rewards vs. Running Average', fontsize=14)

    # Grid, legend, tight layout
    plt.grid(alpha=0.3)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    # Save to file
    plt.savefig(figure_file)
    plt.close()



def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
