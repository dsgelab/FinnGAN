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
from params import *
from train import pretrain_generator, train_GAN

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")
print('Device:', device)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def random_search(n_runs):
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = 30_000_000)

    print('Data loaded, number of individuals:', n_individuals)
    
    # Generator params
    mem_slots = np.arange(1, 21)
    head_size = np.arange(1, 21)
    embed_size = np.arange(2, vocab_size + 1) # Same for the discriminator
    temperature = np.arange(1, 50)
    num_heads = np.arange(1, 21)
    num_blocks = np.arange(1, 21)

    # Discriminator params
    n_embeddings = np.arange(1, vocab_size + 1)
    out_channels = np.arange(1, 21)
    num_filters = np.arange(1, sequence_length - 1)

    # Training params
    batch_size = np.arange(32, 200)
    n_epochs = 8
    print_step = max(n_epochs // 2, 1)
    lr = np.arange(4, 8)
    
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
    
    filename = 'search_results/random_search_{}.csv'.format(n_individuals)
    print(filename)
    
    try:
        resulting_df = pd.read_csv(filename, index_col = 0)
    except FileNotFoundError as e:
        print(e)
        resulting_df = pd.DataFrame()
    print(resulting_df)

    for run in range(n_runs):
        try:
            chosen_params = dict()

            for k, v in params.items():
                chosen_params[k] = int(np.random.choice(v))

            print('Params chosen:', chosen_params)

            mem_slots, head_size, embed_size, temperature, num_heads, num_blocks, n_embeddings, \
                out_channels, num_filters, batch_size, lr = tuple(chosen_params.values())

            filter_sizes = list(range(2, 2 + num_filters)) # values can be at most the sequence_length
            lr = 10 ** (-lr)
            print('lr:', lr)

            dummy_batch_size = 128
            ignore_time = True

            # Train the GAN

            start_time = time.time()

            G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
            D = RelGANDiscriminator(n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes, mbd_out_features, mbd_kernel_dims)

            '''
            if torch.cuda.device_count() > 1:
                print("Using", torch.cuda.device_count(), "GPUs")
                G = nn.DataParallel(G)
                D = nn.DataParallel(D)
            elif cuda:
                print("Using 1 GPU")
            '''

            # Call train function
            scores1, scores2_mean, similarity_score, mode_collapse_score, indv_score_mean, scores2, indv_score, accuracies_real, accuracies_fake = train_GAN(
                G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, print_step, get_scores, ignore_time, dummy_batch_size
            )

            chosen_params['chi-squared_score'] = float(scores1[-1])
            chosen_params['transition_score'] = float(scores2_mean[-1])
            chosen_params['similarity_score'] = float(similarity_score[-1])
            chosen_params['indv_score'] = float(indv_score_mean[-1])
            ser = pd.DataFrame({len(resulting_df): chosen_params}).T
            resulting_df = pd.concat([resulting_df, ser], ignore_index=True)

            print('Time taken:', round_to_n(time.time() - start_time, n = 3), 'seconds')

            print(resulting_df)
            resulting_df.to_csv(filename)
        except RuntimeError as e:
            print(e)

            
def fix_optim_log(filename):
    res = []
    with open(filename) as f:
        content = f.read()
        for iteration in content.split('\n'):
            try:
                obj = json.loads(iteration)
                res.append(obj)
            except json.JSONDecodeError:
                pass
            
    with open(filename, 'w') as outfile:
        json.dump(res, outfile)
    

def optimise(kappa, n_runs, n_sub_runs, ignore_similar, score_type = 'general'):
    n_epochs = 10
    print_step = max(n_epochs // 2, 1)
    
    min_score = -1000
    
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = 10_000_000)
    
    print('Data loaded, number of individuals:', n_individuals)
    
    def objective_function(batch_size, lr, temperature):
        
        try:
            batch_size = int(batch_size)
            lr = 10 ** (-lr)
            
            scores = []
            
            for i in range(n_sub_runs):
                print('sub run {}'.format(i))
                
                # Train the GAN

                G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
                D = RelGANDiscriminator(n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes, use_aux_info, use_mbd, mbd_out_features, mbd_kernel_dims)

                # Call train function
                dist_score, transition_score, similarity_score, mode_collapse_score, indv_score, transition_score_full, _, _, _ = train_GAN(
                    G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, n_critic, print_step, get_scores, ignore_time, dummy_batch_size, ignore_similar, one_sided_label_smoothing, relativistic_average, True
                )
                
                if score_type == 'general':
                    score = -(2 * dist_score[-1] + \
                              1 * transition_score[-1] + \
                              #1 * similarity_score[-1] + \
                              4 * mode_collapse_score[-1])
                elif score_type == 'chd_and_br_cancer':
                    # minimize the transition score from chd to breast cancer
                    score = -transition_score_full[ \
                                  -1, ENDPOINT.vocab.stoi['C3_BREAST'] - 3, ENDPOINT.vocab.stoi['I9_CHD'] - 3 \
                              ]
                elif score_type == 'transition':
                    score = -transition_score[-1]
                    
                if isnan(score):
                    score = min_score
                    
                score = max(min_score, score)
                
                print('Score:', score)
                
                scores.append(score)
                
            score = np.mean(scores)

            return score
    
        except RuntimeError as e:
            print(e)
            return min_score
    
    # Bounded region of parameter space
    pbounds = {
        'batch_size': (16, 512),
        'lr': (2, 8),
        'temperature': (1, 100),
    }

    optimizer = BayesianOptimization(
        f = objective_function,
        pbounds = pbounds,
        #random_state = 1,
    )
    
    filename = "optim_results/{}_{}.json".format(score_type, n_individuals)
    
    load_logs(optimizer, logs=[filename])
    
    logger = JSONLogger(path=filename)
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)        
        
    optimizer.maximize(
        init_points = int(np.sqrt(n_runs)),
        n_iter = n_runs,
    )
    
    #fix_optim_log(filename)
    
    print(optimizer.max)
            
            
if __name__ == '__main__':
    #n_runs = 600
    #with torch.autograd.detect_anomaly():
    #random_search(n_runs)
    
    kappa = 1
    n_runs = 100
    n_sub_runs = 2
    ignore_similar = True
    score_type = 'general'
    
    optimise(kappa, n_runs, n_sub_runs, ignore_similar, score_type)