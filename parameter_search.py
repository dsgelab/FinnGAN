#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import os
import glob
import matplotlib.pyplot as plt
import datetime
from math import ceil, isnan
import sys
import torchtext
from torchtext.data import Field, Iterator, Dataset, Example

from relational_rnn_models import RelationalMemoryGenerator
from discriminator import RelGANDiscriminator
from utils import *
from train import pretrain_generator, train_GAN

cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")
print('Device:', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def random_search(n_runs):
    train, val, ENDPOINT, vocab_size, sequence_length, n_individuals = get_dataset(nrows = 3_000_000)

    print('Data loaded, number of individuals:', n_individuals)
    
    # Generator params
    mem_slots = [1, 2, 5]
    head_size = [1, 2, 5, 10]
    embed_size = [2, 3, 5, 10] # Same for the discriminator
    temperature = [1, 5, 10, 50]
    num_heads = [2, 5, 10, 15]
    num_blocks = [1, 2, 4, 6]

    # Discriminator params
    n_embeddings = [1, 3, 5]
    out_channels = [1, 3, 5, 10]
    num_filters = [1, 2, 3] 

    # Training params
    batch_size = [32, 64, 128]
    n_epochs = 10
    print_step = max(n_epochs // 10, 1)
    lr = [1e-3, 1e-4, 1e-5]
    
    params = dict()
    
    params['mem_slots'] = mem_slots
    params['head_size'] = head_size
    params['embed_size'] = embed_size
    params['temperature'] = temperature
    params['num_heads'] = num_heads
    params['num_blocks'] = num_blocks
    
    params['n_embeddings'] = n_embeddings
    params['out_channels'] = out_channels
    params['num_filters'] = num_filters
    
    params['batch_size'] = batch_size
    params['lr'] = lr
    
    try:
        resulting_df = pd.read_csv('search_results/random_search.csv', index_col = 0)
    except FileNotFoundError as e:
        print(e)
        resulting_df = pd.DataFrame()
    print(resulting_df)

    for run in range(n_runs):
        chosen_params = dict()
        
        for k, v in params.items():
            if k != 'lr':
                chosen_params[k] = int(np.random.choice(v))
            else:
                chosen_params[k] = float(np.random.choice(v))
            
            
        print('Params chosen:', chosen_params)
        
        mem_slots, head_size, embed_size, temperature, num_heads, num_blocks, n_embeddings, \
            out_channels, num_filters, batch_size, lr = tuple(chosen_params.values())
        
        filter_sizes = list(range(2, 2 + num_filters)) # values can be at most the sequence_length
        
        dummy_batch_size = 128

        # Train the GAN

        start_time = time.time()

        G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
        D = RelGANDiscriminator(n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes)

        # Call train function
        scores1, scores2, scores3, accuracies_real, accuracies_fake = train_GAN(
            G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, print_step, get_scores, dummy_batch_size
        )
        
        chosen_params['chi-squared_score'] = float(scores1[-1])
        chosen_params['transition_score'] = float(scores2[-1])
        ser = pd.DataFrame({len(resulting_df): chosen_params}).T
        resulting_df = pd.concat([resulting_df, ser], ignore_index=True)

        print('Time taken:', round_to_n(time.time() - start_time, n = 3), 'seconds')

        
    print(resulting_df)
    resulting_df.to_csv('search_results/random_search.csv')
        


if __name__ == '__main__':
    n_runs = 2
    random_search(n_runs)