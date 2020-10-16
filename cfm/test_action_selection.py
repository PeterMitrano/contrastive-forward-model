import argparse
import pathlib

import torch

from cfm.dataset import DynamicsDataset


# some plotting utilities
def rect(ax, poke, c):
    x, y, t, l, good = poke
    dx = -200 * l * math.cos(t)
    dy = -200 * l * math.sin(t)
    ax.arrow(x, y, dx, dy, head_width=5, head_length=5)


def plot_sample(img_before, img_after, action):
    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img_before.copy())
    rect(ax1, action, "blue")
    ax2.imshow(img_after.copy())


def distance(z_pred, z_next):
    return torch.sum((z_pred - z_next) ** 2, dim=1)


def make_random_actions(n_samples, device):
    r = torch.rand(n_samples, 4).to(device)
    max_action = [0, 0, -1, -1]
    min_action = [64, 64, 1, 1]
    actions = r * (max_action - min_action) + min_action
    return actions


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
    random_actions = make_random_actions(n_samples, device)

    for example in dataset:
        obs, obs_next, true_action = example
        batch_obs = torch.stack(n_samples * [obs]).to(device)
        batch_obs_next = torch.stack(n_samples * [obs_next]).to(device)
        z = encoder(batch_obs)
        z_next = encoder(batch_obs_next)

        z_pred = trans(z, random_actions)

        cost = distance(z_pred, z_next)

        min_idx = torch.argmin(cost)
        best_action = random_actions[min_idx]

        print(best_action)
        print(true_action)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=pathlib.Path)
    parser.add_argument('--dataset', type=pathlib.Path, default=pathlib.Path('data/rope'), help='path to dataset')
    args = parser.parse_args()

    eval(args)


if __name__ == '__main__':
    main()
