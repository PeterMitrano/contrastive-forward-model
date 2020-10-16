import pickle as pkl
from os.path import join

import h5py
import numpy as np
import torch
import torch.utils.data as data


class DynamicsDataset(data.Dataset):
    """
    Dataset that returns current state / next state image pairs
    """

    def __init__(self, root, s=1):
        self.root = root
        self.s = s  # Same 's' as in process_dataset.py (number of timesteps between current / next pairs)

        with open(join(root, f'pos_neg_pairs_{s}.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.pos_pairs = data['pos_pairs']
        self.image_paths = data['all_images']

        # Load entire dataset into memory as opposed to accessing disk with hdf5
        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        # Normalization of actions for different datasets
        if 'rope' in root:
            self.mean = np.array([0.5, 0.5, 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1])
        elif 'cloth' in root:
            self.mean = np.array([0.5, 0.5, 0., 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1, 1])
        else:
            raise Exception('Invalid environment, or environment needed in root name')

    def _get_image(self, path):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        obs_file, obs_next_file, action_file = self.pos_pairs[index]
        obs, obs_next = self._get_image(obs_file), self._get_image(obs_next_file)
        actions = np.load(action_file)

        fsplit = obs_next_file.split('_')
        t = int(fsplit[-2])
        k = int(fsplit[-1].split('.')[0])
        assert self.s == 1
        action = actions[t - 1, k]
        action = (action - self.mean) / self.std

        return obs, obs_next, torch.FloatTensor(action)
