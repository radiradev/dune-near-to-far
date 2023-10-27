import torchvision
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from MinkowskiEngine.utils import sparse_quantize, batch_sparse_collate
import os
# from pytorch3d.transforms import random_rotation

def random_quaternion(device):
    """Generates a random quaternion (w, i, j, k)."""
    u1, u2, u3 = torch.rand(3).to(device)
    u1_sqrt = torch.sqrt(u1)
    u1_sqrt_complement = torch.sqrt(1 - u1)
    u2_double_pi = u2 * 2 * torch.pi
    u3_double_pi = u3 * 2 * torch.pi
    w = torch.cos(u3_double_pi) * u1_sqrt
    x = torch.sin(u2_double_pi) * u1_sqrt_complement
    y = torch.cos(u2_double_pi) * u1_sqrt_complement
    z = torch.sin(u3_double_pi) * u1_sqrt
    return w, x, y, z

def rotate_quaternion(coords, feats, device='cpu'):
    """Rotates coordinates by a random angle using quaternion rotation."""
    w, x, y, z = random_quaternion(device)
    coords = coords.to(device)

    # Convert quaternion into a rotation matrix
    R = torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ]).to(device)

    # Apply rotation to each coordinate
    rotated_coords = torch.matmul(coords, R.t())  # transpose rotation matrix

    return rotated_coords, feats


def drop(coords, feats, p=0.1):
    mask = torch.rand(coords.shape[0]) > p
    return coords[mask], feats[mask]

def shift_adcs(coords, feats, max_scale_factor=0.1):
    shift = 1 - torch.rand(1, dtype=feats.dtype, device=feats.device) * max_scale_factor
    return coords, feats * shift

def translate(coords, feats, cube_size=1000):
    normalized_shift = torch.rand(3, dtype=coords.dtype, device=coords.device)
    translation = normalized_shift * (cube_size / 10)
    return coords + translation, feats


class LarndSimConverted(torch.utils.data.Dataset):
    PIXEL_PITCH = 0.38
    def __init__(
            self,
            root='/pscratch/sd/r/rradev/near_to_far/larndsim_npz_oct27/',
            extensions='.npz',
            augmentations=False,
            targets = ['numu_score', 'nue_score', 'nutau_score', 'nc_score'],
    ):
        super().__init__()
        self.paths = glob(f'{root}/*{extensions}') # use the rest for testing
        self.augmentations = augmentations
        self.targets = targets
        
    def loader(self, path):
        sample = np.load(path)
        coords = sample['coords']
        features = sample['dQ']
        # energy = sample['energy']

        paired_data = sample['paired_data']
        target = np.array([paired_data[col] for col in [self.targets]], dtype=np.float32)
        return coords, features, target


    def preprocessing(self, sample):
        # convert to torch tensors
        coords, features,  target = sample
        coords = torch.from_numpy(coords).float().contiguous()
        features = torch.from_numpy(features).float().unsqueeze(-1)
        target = torch.tensor(np.array([target])).float()
        
        if self.augmentations:
            funcs =  [drop, shift_adcs, translate] 
            funcs_choice = np.random.choice(funcs, 2)
            for func in funcs_choice:
                coords, features = func(coords, features)
        
        
        # quantize the coordinates
        coords, features = sparse_quantize(coords, features=features, quantization_size=self.PIXEL_PITCH)
        return coords, features, target


    def __getitem__(self, index):
        path = self.paths[index]
        sample = self.loader(path)
        sample = self.preprocessing(sample)
        return sample
    
    
    def __len__(self):
        return len(self.paths)

    


if __name__ == '__main__':
    dataset = LarndSimConverted()
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=batch_sparse_collate, drop_last=True)
    for batch in tqdm(dataloader):
        coords, feats, labels = batch
        assert coords.shape[0] > 0 and coords.shape[1] == 4 and feats.shape[1] == 2 and labels.shape[1] == 4 and labels.shape[0] == batch_size, f'coords: {coords.shape}, feats: {feats.shape}, labels: {labels.shape}'

