import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
import os
import math
import datetime
import time
from frechetdist import frdist

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch

from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from models import Generator, Discriminator
from data.sparse_molecular_dataset import SparseMolecularDataset
from rdkit import Chem


def str2bool(v):
    return v.lower() in ('true')

qubits = 2
# Set up your ibmq credentials first from https://quantum-computing.ibm.com/
demo_on_ibmq = True
#here we are using default device provided by pennylane
if demo_on_ibmq:
    dev = qml.device('qiskit.ibmq', wires=2, backend='ibmq_manila',
                 ibmqx_token='c87ce3ffb49c6ed9da89699416a6f26df7b887eefeb0e0e45048417e260ebd184078c73657eb357a31214d016cb1c562133f142714723da220f8b302e918b88f', hub='ibm-q-education', group='cornell-uni-2', project='fa22-phys-4481')
else:
    dev = qml.device('default.qubit', wires=qubits)

@qml.qnode(dev, interface='torch')
def gen_circuit_1(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 2
    
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

@qml.qnode(dev, interface='torch')
def gen_circuit_2(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 2
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i+3], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits+3], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

@qml.qnode(dev, interface='torch')
def gen_circuit_3(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 2
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i+3*2], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits+3*2], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

@qml.qnode(dev, interface='torch')
def gen_circuit_4(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    layers = 1
    qubits = 2
    
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layers):
        for i in range(qubits):
            qml.RY(w[i+3*3], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits+3*3], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Solver for training and testing StarGAN.
    self = Solver(config)
    from logger import Logger
    self.logger = Logger(self.log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

        # Quantum circuit configuration
    parser.add_argument('--quantum', type=bool, default=True, help='choose to use quantum gan with hybrid generator')
    parser.add_argument('--patches', type=int, default=1, help='number of quantum circuit patches')
    parser.add_argument('--layer', type=int, default=1, help='number of repeated variational quantum layer')
    parser.add_argument('--qubits', type=int, default=8, help='number of qubits and dimension of domain labels')
    
    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[128], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=5000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=2500, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=5000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Directories.
    parser.add_argument('--mol_data_dir', type=str, default='data/gdb9_9nodes.sparsedataset')
    parser.add_argument('--log_dir', type=str, default='p4_qgan_hg_15p/logs')
    parser.add_argument('--model_save_dir', type=str, default='p4_qgan_hg_15p/models')
    parser.add_argument('--sample_dir', type=str, default='p4_qgan_hg_15p/samples')
    parser.add_argument('--result_dir', type=str, default='p4_qgan_hg_15p/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=500)

    config = parser.parse_args()
    print(config)
    main(config)
