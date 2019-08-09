import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field, Iterator, Dataset, Example
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
    
    return 
    
    # TODO: change to as they are in get_scores
    if ignore_similar:
        similarity_score = torch.tensor(1.0 - data_fake.shape[0] / data.shape[0])
    else:
        similarity_score = robust_get_similarity_score(data, data_fake, dummy_batch_size2, False)
        

    score1 = get_score(data_fake, ENDPOINT, vocab_size)
    
    transition_score = get_aggregate_transition_score(data, data_fake, ignore_time, True, True, vocab_size, sequence_length)
    
    indv_score = get_individual_score(data, data_fake, True, vocab_size, sequence_length)
    
    print(score1, transition_score.mean(), similarity_score, indv_score.mean(), transition_score, indv_score)
    

    
    
def simulate_training():
    nrows = 300_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    
    print('Data loaded, number of individuals:', n_individuals)
    
    print('GAN type:', GAN_type)
    print('Relativistic average:', relativistic_average)

    # Train the GAN

    
    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    
    
    N_max = 10
    prefix = 'Before:'
    
    G.eval()
    
    data1, ages1, sexes1, data_fake1, ages_fake1, sexes_fake1 = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, True)
    
    data2, ages2, sexes2, data_fake2, ages_fake2, sexes_fake2 = get_real_and_fake_data(G, val, ignore_similar, dummy_batch_size, sequence_length, True)
    
    save_frequency_comparisons(data_fake1, data_fake2, vocab_size, ENDPOINT, prefix, N_max)   
    
    event_name = 'I9_CHD'
    predictor_name = 'C3_BREAST'
    
    analyse(data1, data_fake1, True, True, ENDPOINT, sequence_length, event_name, predictor_name)
    test_generator(data1, ages1, sexes1, data_fake1, ages_fake1, sexes_fake1, True, True, ENDPOINT, SEX, vocab_size, sequence_length)
    
    
    analyse(data2, data_fake2, True, False, ENDPOINT, sequence_length, event_name, predictor_name)
    test_generator(data2, ages2, sexes2, data_fake2, ages_fake2, sexes_fake2, True, False, ENDPOINT, SEX, vocab_size, sequence_length)


    G.load_state_dict(torch.load(G_filename))
    
    
    prefix = 'After:'
    
    data1, ages1, sexes1, data_fake1, ages_fake1, sexes_fake1 = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, True)
    
    data2, ages2, sexes2, data_fake2, ages_fake2, sexes_fake2 = get_real_and_fake_data(G, val, ignore_similar, dummy_batch_size, sequence_length, True)

    save_frequency_comparisons(data_fake1, data_fake2, vocab_size, ENDPOINT, prefix, N_max)    
    


    save_plots_of_train_scores(scores1_train, transition_scores_mean_train, similarity_score_train, indv_score_mean_train, transition_scores_train, indv_score_train, \
    scores1_val, transition_scores_mean_val, similarity_score_val, indv_score_mean_val, transition_scores_val, indv_score_val, \
    accuracies_real, accuracies_fake, ignore_time, sequence_length, vocab_size, ENDPOINT)


    test_size = 10
    visualize_output(G, test_size, val, sequence_length, ENDPOINT, SEX)
    
    analyse(data1, data_fake1, False, True, ENDPOINT, sequence_length, event_name, predictor_name)
    test_generator(data1, ages1, sexes1, data_fake1, ages_fake1, sexes_fake1, False, True, ENDPOINT, SEX, vocab_size, sequence_length)
    
    
    analyse(data2, data_fake2, False, False, ENDPOINT, sequence_length, event_name, predictor_name)
    test_generator(data2, ages2, sexes2, data_fake2, ages_fake2, sexes_fake2, False, False, ENDPOINT, SEX, vocab_size, sequence_length)
    
    save(data1, data_fake1, train = True)
    save(data2, data_fake2, train = False)

    
    
    
    
    
if __name__ == '__main__':

    a = torch.randn(2, 4, 3)
    print(a)
    print()
    print(a.view(a.shape[0], -1))
    b = torch.cat([a, torch.ones(a.shape[:2]).unsqueeze(-1)], dim = -1)
    print(b)
    
    '''
    
    nrows = 300_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    print('Data loaded, number of individuals:', n_individuals)
    
    data = next(iter(Iterator(train, batch_size = n_individuals))).ENDPOINT.transpose(0, 1)
    print(1.0 - data.unique(dim = 0).shape[0] / data.shape[0]) # 0.771352528429712 (nrows = 300_000_000)
    
    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    G.load_state_dict(torch.load(G_filename))
    G.eval()
    
    data, ages, sexes, data_fake, ages_fake, sexes_fake = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, True)
    
    test_generator(data, ages, sexes, data_fake, ages_fake, sexes_fake, False, True, ENDPOINT, SEX, vocab_size, sequence_length)
    
    simulate_training()
    
    '''
    