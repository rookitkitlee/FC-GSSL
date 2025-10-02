import os
import argparse
import math
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from load_dataset import MoleculeDataset
from dataloader import collate_fn
from utils import load_config, create_optimizer

from autoencoder import GraphAutoEncoder
from encoder import GraphEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument('--root', type=str, default="../dataset")
parser.add_argument('--dir_name', type=str, default="./mask_atom_noise_pe")
parser.add_argument("--dataset", type=str, default="zinc_standard_agent")  # zinc_standard_agent
args = parser.parse_args()

config = load_config(f"./config/pretraining_on_zinc.yaml")
for key, value in config.items():
    setattr(args, key, value)

processed_name = f"processed_{args.lap_norm}"

print('main-------------------------1')
dataset = MoleculeDataset(root=args.root, dataset=args.dataset, max_freqs=args.max_freqs, lap_norm=args.lap_norm, processed_name=processed_name)
print('main-------------------------2')


torch.save(dataset, f'../dataset/x_zinc.pt')
