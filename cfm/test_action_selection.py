import argparse
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm.dataset import DynamicsDataset


# some plotting utilities
def rect(ax, poke, c):
    x, y, clockwise_angle, l = poke
    dx = -200 * l * math.cos(clockwise_angle) * (64 / 240)
    dy = -200 * l * math.sin(clockwise_angle) * (64 / 240)
    ax.arrow(x * (64 / 240), y * (64 / 240), dx, dy, head_width=5, head_length=5, color=c)


def distance(z_pred, z_next):
    return torch.sum((z_pred - z_next) ** 2, dim=1)


def make_random_actions(n_samples, device):
    r = torch.rand(n_samples, 4).to(device)
    max_action = torch.tensor([0., 0, 0, 0.01]).to(device)
    min_action = torch.tensor([240., 240, 2 * np.pi, 0.15]).to(device)
    actions = r * (max_action - min_action) + min_action
    return actions


def to_plt_img(x):
    return x.transpose(0, 2).cpu().rot90().numpy()[::-1, :]
    # return x.transpose(0, 2).cpu().numpy()[:, ::-1]


def eval(args):
    dataset = DynamicsDataset(root=(args.dataset / 'test_data').as_posix())

    device = torch.device('cuda')
    checkpoint = torch.load(args.checkpoint.as_posix(), map_location=device)

    all_actions = []
    for d in dataset:
        all_actions.append(d[2])
    all_actions = torch.stack(all_actions)
    torch.min(all_actions, dim=0)
    torch.max(all_actions, dim=0)

    encoder = checkpoint['encoder']
    trans = checkpoint['trans']

    n_samples = 100

    for example in dataset:
        obs, obs_next, true_action = example
        batch_obs = torch.stack(n_samples * [obs]).to(device)
        batch_obs_next = torch.stack(n_samples * [obs_next]).to(device)
        z = encoder(batch_obs)
        z_next = encoder(batch_obs_next)

        random_actions = make_random_actions(n_samples - 1, device)
        random_actions = torch.cat([random_actions, torch.unsqueeze(true_action, 0).to(device)])
        z_pred = trans(z, random_actions)

        cost = distance(z_pred, z_next)

        min_idx = torch.argmin(cost)

        best_action = random_actions[min_idx]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(to_plt_img(obs))
        rect(ax1, best_action.cpu(), "#0000ff88")
        rect(ax1, true_action.cpu(), "#ff0000aa")
        ax2.imshow(to_plt_img(obs_next))
        plt.show()

        print(best_action)
        print(true_action)


def main():
    parser = argparse.ArgumentParser()
    np.set_printoptions(suppress=True, precision=3)

    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--dataset', type=pathlib.Path, default=pathlib.Path('data/rope'), help='path to dataset')
    args = parser.parse_args()

    eval(args)


if __name__ == '__main__':
    main()
