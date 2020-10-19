import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from cfm.dataset import DynamicsDataset


def unnormalize_action(dataset, action):
    partially_unnormalized = (action.cpu() * dataset.std + dataset.mean)
    x, y, dx, dy = partially_unnormalized
    # checked with the original authors code, and yes we do need to flip y and swap x/y
    return (x * 63) * 1.0, (y * 63) * 1.0, -dy * 63 / 2, dx * 63 / 2


# some plotting utilities
def rect(ax, action, c):
    x, y, dx, dy = action
    ax.arrow(x, y, dx, dy, head_width=5, head_length=5, color=c)


def distance(z_pred, z_next):
    return torch.sum((z_pred - z_next) ** 2, dim=1)


def make_random_actions(n_samples, device):
    r = torch.rand(n_samples, 4).to(device)
    max_action = torch.tensor([-1., -1, -1, -1]).to(device)
    min_action = torch.tensor([1., 1, 1, 1]).to(device)
    actions = r * (max_action - min_action) + min_action
    return actions


def to_plt_img(x):
    # return x.transpose(0, 2).cpu().rot90().numpy()[::-1, :]
    return np.clip(x.transpose(0, 2).cpu(), 0, 1)


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

    n_samples = 1000
    action_errors = []
    include_true_action = False

    for example_idx, example in enumerate(dataset):
        if example_idx > 100:
            break
        obs, obs_next, true_action = example
        batch_obs = torch.stack(n_samples * [obs]).to(device)
        batch_obs_next = torch.stack(n_samples * [obs_next]).to(device)
        z = encoder(batch_obs)
        z_next = encoder(batch_obs_next)

        if include_true_action:
            random_actions = make_random_actions(n_samples - 1, device)
            random_actions = torch.cat([random_actions, torch.unsqueeze(true_action, 0).to(device)])
        else:
            random_actions = make_random_actions(n_samples, device)
        z_pred = trans(z, random_actions)

        cost = distance(z_pred, z_next)

        min_idx = torch.argmin(cost)

        best_action = random_actions[min_idx]

        if args.plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.set_xlim(0, 64)
            ax1.set_ylim(0, 64)
            ax2.set_xlim(0, 64)
            ax2.set_ylim(0, 64)
            ax1.imshow(to_plt_img(obs))
            best_action_unnormalized = unnormalize_action(dataset, best_action)
            true_action_unnormalized = unnormalize_action(dataset, true_action)
            rect(ax1, best_action_unnormalized, "#0000ff88")
            rect(ax1, true_action_unnormalized, "#ff0000aa")
            ax2.imshow(to_plt_img(obs_next))
            plt.show()

        action_error = torch.norm(best_action.cpu() - true_action.cpu())
        action_errors.append(action_error)

    action_errors = np.array(action_errors)
    print("overall average error in prediction actions {:.3f}".format(action_errors.mean()))


def main():
    parser = argparse.ArgumentParser()
    np.set_printoptions(suppress=True, precision=3)

    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--dataset', type=pathlib.Path, default=pathlib.Path('data/rope_dr_frame'), help='path to dataset')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    eval(args)


if __name__ == '__main__':
    main()
