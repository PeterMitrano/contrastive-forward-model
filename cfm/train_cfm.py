import argparse
import json
import os
import time
from os.path import join, exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import cfm.models as cm
import cfm.utils as cu
from cfm.dataset import DynamicsDataset


def get_dataloaders():
    train_dset = DynamicsDataset(root=join(args.root, 'train_data'))
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4,
                                   pin_memory=True)

    test_dset = DynamicsDataset(root=join(args.root, 'test_data'))
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  num_workers=4, pin_memory=True)

    return train_loader, test_loader


def compute_cpc_loss(obs, obs_pos, encoder, trans, actions, device):
    bs = obs.shape[0]

    z, z_pos = encoder(obs), encoder(obs_pos)  # b x z_dim
    z_next = trans(z, actions)

    neg_dot_products = torch.mm(z_next, z.t())  # b x b
    neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2 * neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    idxs = np.arange(bs)
    # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
    neg_dists[idxs, idxs] = float('-inf')  # b x b

    pos_dot_products = (z_pos * z_next).sum(dim=1)  # b
    pos_dists = -((z_pos ** 2).sum(1) - 2 * pos_dot_products + (z_next ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1)  # b x 1

    dists = torch.cat((neg_dists, pos_dists), dim=1)  # b x b+1
    dists = F.log_softmax(dists, dim=1)  # b x b+1
    loss = -dists[:, -1].mean()  # Get last column with is the true pos sample

    return loss


def train(encoder, trans, optimizer, train_loader, epoch, device):
    encoder.train()
    trans.train()

    stats = cu.Stats()
    pbar = tqdm(total=len(train_loader.dataset))

    parameters = list(encoder.parameters()) + list(trans.parameters())
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]
        loss = compute_cpc_loss(obs, obs_pos, encoder,
                                trans, actions, device)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()
    return stats


def test(encoder, trans, test_loader, epoch, device):
    encoder.eval()
    trans.eval()

    test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            loss = compute_cpc_loss(obs, obs_pos, encoder,
                                    trans, actions, device)
            test_loss += loss * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = 'out/rope_' + str(int(time.time()))
    if not exists(folder_name):
        os.makedirs(folder_name)

    writer = SummaryWriter(folder_name)

    save_args = vars(args)
    save_args['script'] = 'train_cfm'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(save_args, f)

    obs_dim = (3, 64, 64)
    if 'rope' in args.root:
        action_dim = 4
    elif 'cloth' in args.root:
        action_dim = 5
    else:
        raise Exception('Invalid environment, or environment needed in root name')

    device = torch.device('cuda')

    encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
    trans = cm.Transition(args.z_dim, action_dim, trans_type=args.trans_type).to(device)
    parameters = list(encoder.parameters()) + list(trans.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    print(folder_name)

    if args.load_checkpoint:
        checkpoint = torch.load(join(folder_name, 'checkpoint'))
        encoder.load_state_dict(checkpoint['encoder'])
        trans.load_state_dict(checkpoint['trans'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    train_loader, test_loader = get_dataloaders()

    # Save example training images
    batch = next(iter(train_loader))
    obs, obs_next, _ = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    example_images_path = join(folder_name, 'train_seq_img.png')
    print(example_images_path)
    cu.save_image(imgs * 0.5 + 0.5, example_images_path, nrow=8)

    best_test_loss = float('inf')
    itr = 0

    # save once before any training
    save_checkpoint(best_test_loss, encoder, folder_name, optimizer, trans)

    for epoch in range(args.epochs):
        # Train
        stats = train(encoder, trans, optimizer, train_loader, epoch, device)
        test_loss = test(encoder, trans, test_loader, epoch, device)

        # Log metrics
        old_itr = itr
        for k, values in stats.items():
            itr = old_itr
            for v in values:
                writer.add_scalar(k, v, itr)
                itr += 1
        writer.add_scalar('test_loss', test_loss, epoch)

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                save_checkpoint(best_test_loss, encoder, folder_name, optimizer, trans)
    writer.close()


def save_checkpoint(best_test_loss, encoder, folder_name, optimizer, trans):
    checkpoint = {
        'encoder': encoder,
        'trans': trans,
        'optimizer': optimizer,
    }
    checkpoint_file = join(folder_name, 'checkpoint')
    torch.save(checkpoint, checkpoint_file)
    print('Saved {} with loss {}'.format(checkpoint_file, best_test_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/rope', help='path to dataset (default: data/rope)')

    # Architecture Parameters
    parser.add_argument('--trans_type', type=str, default='reparam_w_tanh',
                        help='linear | mlp | reparam_w | reparam_w_ortho_gs | reparam_w_ortho_cont | reparam_w_tanh (default: linear)')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='default 0')
    parser.add_argument('--epochs', type=int, default=30, help='default: 50')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--load_checkpoint', action='store_true')

    # InfoNCE Parameters
    # negative samples are the other batch elements, so number of negative samples
    # is the same as the batch size
    parser.add_argument('--batch_size', type=int, default=128, help='default 128')
    parser.add_argument('--z_dim', type=int, default=32, help='dimension of the latents')

    # Other
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    assert args.trans_type in ['linear', 'mlp', 'reparam_w', 'reparam_w_ortho_gs', 'reparam_w_ortho_cont', 'reparam_w_tanh']
    main()
