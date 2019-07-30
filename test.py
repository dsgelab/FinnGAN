import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from params import *
from utils import *
from relational_rnn_models import RelationalMemoryGenerator


def test_association(G, train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals):
    
    data, ages, sexes, data_fake, ages_fake, sexes_fake = get_real_and_fake_data(G, train, ignore_similar, 128, sequence_length, True)
    
    subjects_with_br_cancer = (data_fake == ENDPOINT.vocab.stoi['C3_BREAST']).any(dim = 1)
    sexes_of_subjects_with_br_cancer = sexes_fake[subjects_with_br_cancer]
    association_score = (sexes_of_subjects_with_br_cancer == SEX.vocab.stoi['female']).float().mean()
    
    print('Breast cancer - sex association score:', association_score)
    
    
    _, transition_freq = get_transition_matrix(data, vocab_size, None, ignore_time)
    from_br_cancer_to_chd = transition_freq[ENDPOINT.vocab.stoi['C3_BREAST'] - 3, ENDPOINT.vocab.stoi['I9_CHD'] - 3, 1]
    
    _, transition_freq_fake = get_transition_matrix(data_fake, vocab_size, None, ignore_time)
    from_br_cancer_to_chd_fake = transition_freq_fake[ENDPOINT.vocab.stoi['C3_BREAST'] - 3, ENDPOINT.vocab.stoi['I9_CHD'] - 3, 1]
    
    print('Freq from breast cancer to CHD (real):', from_br_cancer_to_chd)
    print('Freq from breast cancer to CHD (fake):', from_br_cancer_to_chd_fake)
    
    return 
    print()
    print(transition_freq[:, :, 1])
    print()
    print(transition_freq_fake[:, :, 1])
    print()
    
    print(get_scores(G, ENDPOINT, train, 128, ignore_time, True, True, ignore_similar, vocab_size, sequence_length))
    print(get_scores(G, ENDPOINT, val, 128, ignore_time, True, True, ignore_similar, vocab_size, sequence_length))
    


if __name__ == '__main__':
    
    GAN_type = 'feature matching'

    relativistic_average = False

    params_name = 'general'

    G_filename = 'models/{}_{}_{}_model.pt'.format(params_name, GAN_type, relativistic_average)
    
    nrows = 30_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    print('Data loaded, number of individuals:', n_individuals)
    
    if params_name == 'general':
        parameters = general_params['params']
    elif params_name == 'br_cancer_and_chd':
        parameters = br_cancer_and_chd_params['params']
    
    embed_size = parameters['embed_size']
    head_size = parameters['head_size']
    mem_slots = parameters['mem_slots']
    num_blocks = parameters['num_blocks']
    num_heads = parameters['num_heads']
    temperature = parameters['temperature']
    
    embed_size = int(embed_size)
    head_size = int(head_size)
    mem_slots = int(mem_slots)
    num_blocks = int(num_blocks)
    num_heads = int(num_heads)

    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    G.load_state_dict(torch.load(G_filename))
    
    test_association(G, train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals)