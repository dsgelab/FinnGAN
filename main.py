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
from survival_analysis import analyse
from test import test_association

cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")
print('Device:', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def main():
    if params_name == 'general':
        parameters = general_params['params']
    elif params_name == 'br_cancer_and_chd':
        parameters = br_cancer_and_chd_params['params']
    
    batch_size = parameters['batch_size']
    embed_size = parameters['embed_size']
    head_size = parameters['head_size']
    lr = parameters['lr']
    mem_slots = parameters['mem_slots']
    n_embeddings = parameters['n_embeddings']
    num_blocks = parameters['num_blocks']
    num_filters = parameters['num_filters']
    num_heads = parameters['num_heads']
    out_channels = parameters['out_channels']
    temperature = parameters['temperature']
    n_critic = parameters['n_critic']
    
    batch_size = int(batch_size)
    embed_size = int(embed_size)
    head_size = int(head_size)
    lr = int(lr)
    mem_slots = int(mem_slots)
    n_embeddings = int(n_embeddings)
    num_blocks = int(num_blocks)
    num_filters = int(num_filters)
    num_heads = int(num_heads)
    out_channels = int(out_channels)
    n_critic = int(n_critic)

    filter_sizes = list(range(2, 2 + num_filters)) # values can be at most the sequence_length
    lr = 10 ** (-lr)

    
    nrows = 300_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    
    print('Data loaded, number of individuals:', n_individuals)
    
    print('GAN type:', GAN_type)
    print('Relativistic average:', relativistic_average)

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
    
    G.eval()
    save_frequency_comparisons(G, train, val, dummy_batch_size, vocab_size, sequence_length, ENDPOINT, prefix, N_max)
    G.train()

    # Call train function
    scores1_train, transition_scores_mean_train, similarity_score_train, indv_score_mean_train, transition_scores_train, indv_score_train, \
    scores1_val, transition_scores_mean_val, similarity_score_val, indv_score_mean_val, transition_scores_val, indv_score_val, \
    accuracies_real, accuracies_fake = train_GAN(
        G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, n_critic, print_step, get_scores, ignore_time, dummy_batch_size, ignore_similar, one_sided_label_smoothing, relativistic_average
    )

    G.eval()
    
    prefix = 'After:'

    save_frequency_comparisons(G, train, val, dummy_batch_size, vocab_size, sequence_length, ENDPOINT, prefix, N_max)
    

    print('Time taken:', round_to_n(time.time() - start_time, n = 3), 'seconds')

    save_plots_of_train_scores(scores1_train, transition_scores_mean_train, similarity_score_train, indv_score_mean_train, transition_scores_train, indv_score_train, \
    scores1_val, transition_scores_mean_val, similarity_score_val, indv_score_mean_val, transition_scores_val, indv_score_val, \
    accuracies_real, accuracies_fake, ignore_time, sequence_length, vocab_size, ENDPOINT)


    test_size = 10
    visualize_output(G, test_size, val, sequence_length, ENDPOINT, SEX)
    
    torch.save(G.state_dict(), G_filename)
    
    event_name = 'I9_CHD'
    predictor_name = 'C3_BREAST'
    analyse(G, train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals, event_name, predictor_name)
    
    test_association(G, train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals)


if __name__ == '__main__':
    main()