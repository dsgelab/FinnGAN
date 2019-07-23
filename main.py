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
from params import *
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

def main():
    
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = 30_000_000)
    
    print('Data loaded, number of individuals:', n_individuals)

    # Train the GAN

    start_time = time.time()

    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    D = RelGANDiscriminator(n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes)
    
    '''
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
    elif cuda:
        print("Using 1 GPU")
    '''
        
    N_max = 10
    prefix = 'Before:'
    
    save_frequency_comparisons(G, train, val, dummy_batch_size, vocab_size, sequence_length, ENDPOINT, prefix, N_max)

    # Call train function
    scores1, scores2_mean, similarity_score, indv_score_mean, scores2, indv_score, accuracies_real, accuracies_fake = train_GAN(
        G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, print_step, get_scores, ignore_time, dummy_batch_size
    )

    prefix = 'After:'

    save_frequency_comparisons(G, train, val, dummy_batch_size, vocab_size, sequence_length, ENDPOINT, prefix, N_max)
    

    print('Time taken:', round_to_n(time.time() - start_time, n = 3), 'seconds')

    save_plots_of_train_scores(scores1, scores2_mean, similarity_score, indv_score_mean, scores2, indv_score, accuracies_real, accuracies_fake, ignore_time, sequence_length, vocab_size, ENDPOINT)


    test_size = 10
    visualize_output(G, test_size, val, sequence_length, ENDPOINT, SEX)
    
    
    filename = 'models/model.pt'
    torch.save(G.state_dict(), filename)


if __name__ == '__main__':
    main()