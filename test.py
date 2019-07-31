import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from params import *
from utils import *
from relational_rnn_models import RelationalMemoryGenerator
    
def plot_transition_matrix_comparisons(transition_freq_real, transition_freq_fake, before, train, ENDPOINT, vocab_size, title):
    plt.style.use('classic')
    
    fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
    fig.subplots_adjust(left=0.22075, right=0.9)
    ticks = np.arange(vocab_size - 3)
    labels = [ENDPOINT.vocab.itos[i + 3] for i in ticks]
    cmap = 'plasma'
    
    vmax = torch.max(transition_freq_fake.max(), transition_freq_real.max())
    
    im = ax[0].matshow(transition_freq_real, vmin=0, vmax=vmax, cmap=cmap)
    ax[0].set_xticks(ticks)
    ax[0].set_xticklabels(labels, rotation=90)
    ax[0].set_title('Real', y = -0.2)
    
    ax[1].matshow(transition_freq_fake, vmin=0, vmax=vmax, cmap=cmap)
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(labels, rotation=90)
    ax[1].set_title('Fake', y = -0.2)
    
    ax[2].matshow(torch.abs(transition_freq_fake - transition_freq_real), vmin=0, vmax=vmax, cmap=cmap)
    ax[2].set_xticks(ticks)
    ax[2].set_xticklabels(labels, rotation=90)
    ax[2].set_title('Abs. difference', y = -0.2)
    
    plt.yticks(ticks, labels)
    
    fig.colorbar(im, ax=ax.ravel().tolist(), ticks=np.linspace(0, vmax, 5), shrink = 0.27, aspect = 10)
    if title:
        fig.suptitle('Transition probabilities ({}, {})'.format('Before' if before else 'After', 'train' if train else 'val'))
    fig.savefig('figs/{}_transition_matrices_{}.svg'.format('Before' if before else 'After', 'train' if train else 'val'))
    
    plt.style.use(plot_style)

def test_generator(data, ages, sexes, data_fake, ages_fake, sexes_fake, before, train, ENDPOINT, SEX, vocab_size, sequence_length):
    subjects_with_br_cancer = (data_fake == ENDPOINT.vocab.stoi['C3_BREAST']).any(dim = 1)
    sexes_of_subjects_with_br_cancer = sexes_fake[subjects_with_br_cancer]
    association_score = (sexes_of_subjects_with_br_cancer == SEX.vocab.stoi['female']).float().mean()
    
    print('Breast cancer - sex association score ({}, {}):'.format('Before' if before else 'After', 'train' if train else 'val'), association_score)
    
    
    transition_freq = get_transition_matrix(data, vocab_size, None, ignore_time)
    from_br_cancer_to_chd = transition_freq[ENDPOINT.vocab.stoi['C3_BREAST'] - 3, ENDPOINT.vocab.stoi['I9_CHD'] - 3]
    
    transition_freq_fake = get_transition_matrix(data_fake, vocab_size, None, ignore_time)
    from_br_cancer_to_chd_fake = transition_freq_fake[ENDPOINT.vocab.stoi['C3_BREAST'] - 3, ENDPOINT.vocab.stoi['I9_CHD'] - 3]
    
    print('Freq from breast cancer to CHD (real):', from_br_cancer_to_chd)
    print('Freq from breast cancer to CHD (fake):', from_br_cancer_to_chd_fake)
    
    plot_transition_matrix_comparisons(transition_freq, transition_freq_fake, before, train, ENDPOINT, vocab_size, True)
    
    #return 
    
    if ignore_similar:
        similarity_score = torch.tensor(1.0 - data_fake.shape[0] / data.shape[0])
    else:
        similarity_score = robust_get_similarity_score(data, data_fake, batch_size, False)
        

    score1 = get_score(data_fake, ENDPOINT, vocab_size)
    
    transition_score = get_aggregate_transition_score(data, data_fake, ignore_time, True, True, vocab_size, sequence_length)
    
    indv_score = get_individual_score(data, data_fake, True, vocab_size, sequence_length)
    
    print(score1, transition_score.mean(), similarity_score, indv_score.mean(), transition_score, indv_score)
    


if __name__ == '__main__':
    nrows = 3_000_000
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
    G.eval()
    
    data, ages, sexes, data_fake, ages_fake, sexes_fake = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, True)
    
    test_generator(data, ages, sexes, data_fake, ages_fake, sexes_fake, False, True, ENDPOINT, SEX, vocab_size, sequence_length)