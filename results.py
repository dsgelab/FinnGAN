import torch
import numpy as np
from utils import *
from params import *
from train import *
import os
import json
from relational_rnn_models import RelationalMemoryGenerator
from discriminator import RelGANDiscriminator
from survival_analysis import analyse
from lifelines.utils import ConvergenceError



def main(use_aux_info, use_mbd, n_endpoints, n_runs):
    endpoints_used = endpoints[:n_endpoints] 
    
    args = {
        'use_aux_info': use_aux_info,
        'use_mbd': use_mbd,
        'n_endpoints': n_endpoints,
    }
    
    dirname_parts = []
    
    for k, v in args.items():
        dirname_parts.append(k + '=' + str(v))
    
    dirname = 'results/' + ' '.join(dirname_parts) + '/'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
            
    nrows = 300_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows, endpoints_used)

    print('Data loaded, number of individuals:', n_individuals)
        
    for seed in range(n_runs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        subdir = dirname + str(seed) + '/'
        
        if not os.path.exists(subdir):
            print('Run {}'.format(seed))
            
            G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
            D = RelGANDiscriminator(n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes, use_aux_info, use_mbd, mbd_out_features, mbd_kernel_dims)
            
            scores1_train, transition_scores_mean_train, similarity_score1_train, similarity_score2_train, mode_collapse_score_train, \
    scores1_val, transition_scores_mean_val, similarity_score1_val, similarity_score2_val, mode_collapse_score_val, \
    accuracies_real, accuracies_fake = train_GAN(
        G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, n_critic, print_step, get_scores, ignore_time, dummy_batch_size, ignore_similar, one_sided_label_smoothing, relativistic_average, False
    )
            
            os.mkdir(subdir)
            
            torch.save(scores1_train, subdir + 'chi-sqrd_train.pt')
            torch.save(transition_scores_mean_train, subdir + 'transition_train.pt')
            torch.save(similarity_score1_train, subdir + 'similarity1_train.pt')
            torch.save(similarity_score2_train, subdir + 'similarity2_train.pt')
            torch.save(mode_collapse_score_train, subdir + 'mode_collapse_train.pt')
            torch.save(scores1_val, subdir + 'chi-sqrd_val.pt')
            torch.save(transition_scores_mean_val, subdir + 'transition_val.pt')
            torch.save(similarity_score1_val, subdir + 'similarity1_val.pt')
            torch.save(similarity_score2_val, subdir + 'similarity2_val.pt')
            torch.save(mode_collapse_score_val, subdir + 'mode_collapse_val.pt')
            torch.save(accuracies_real, subdir + 'acc_real.pt')
            torch.save(accuracies_fake, subdir + 'acc_fake.pt')
            
            
            G.eval()
        
            data1, ages1, sexes1, data_fake1, ages_fake1, sexes_fake1 = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, True)
    
            data2, ages2, sexes2, data_fake2, ages_fake2, sexes_fake2 = get_real_and_fake_data(G, val, ignore_similar, dummy_batch_size, sequence_length, True)

            for pred_i in range(3, vocab_size):
                for event_i in range(3, vocab_size):
                    if pred_i != event_i:
                        predictor_name = ENDPOINT.vocab.itos[pred_i]
                        event_name = ENDPOINT.vocab.itos[event_i]
                        
                        fname_base = subdir + predictor_name + '->' + event_name
                        
                        try:
                            df_train, hr1_train, ci1_train, hr2_train, ci2_train = analyse(data1, data_fake1, False, True, ENDPOINT, event_name, predictor_name, plot=False)

                            df_train.to_csv(fname_base + '_train.csv')
                            hr1_train.to_csv(fname_base + '_hr_real_train.csv')
                            ci1_train.to_csv(fname_base + '_ci_real_train.csv')
                            hr2_train.to_csv(fname_base + '_hr_fake_train.csv')
                            ci2_train.to_csv(fname_base + '_ci_fake_train.csv')
                        except ConvergenceError:
                            pass
                        
                        try:
                            df_val, hr1_val, ci1_val, hr2_val, ci2_val = analyse(data2, data_fake2, False, False, ENDPOINT, event_name, predictor_name, plot=False)

                            df_val.to_csv(fname_base + '_val.csv')
                            hr1_val.to_csv(fname_base + '_hr_real_val.csv')
                            ci1_val.to_csv(fname_base + '_ci_real_val.csv')
                            hr2_val.to_csv(fname_base + '_hr_fake_val.csv')
                            ci2_val.to_csv(fname_base + '_ci_fake_val.csv')
                        except ConvergenceError:
                            pass


            freqs, freqs_fake, counts, counts_fake = save_frequency_comparisons(data_fake1, data_fake2, vocab_size, ENDPOINT, '', 10, plot=False)
            
            torch.save(freqs, subdir + 'freqs.pt')
            torch.save(freqs_fake, subdir + 'freqs_fake.pt')
            torch.save(counts, subdir + 'counts.pt')
            torch.save(counts_fake, subdir + 'counts_fake.pt')


            transition_freq = get_transition_matrix(data1, vocab_size, None, ignore_time)
            transition_freq_fake = get_transition_matrix(data_fake1, vocab_size, None, ignore_time)
        
            torch.save(transition_freq, subdir + 'transition_matrix_real_train.pt')
            torch.save(transition_freq_fake, subdir + 'transition_matrix_fake_train.pt')
            
            transition_freq = get_transition_matrix(data2, vocab_size, None, ignore_time)
            transition_freq_fake = get_transition_matrix(data_fake2, vocab_size, None, ignore_time)
        
            torch.save(transition_freq, subdir + 'transition_matrix_real_val.pt')
            torch.save(transition_freq_fake, subdir + 'transition_matrix_fake_val.pt')
        
        else:
            print('Skipping run {},'.format(seed), subdir, 'already exists')
        



if __name__ == '__main__':
    
    use_aux_info = True 
    use_mbd = True
    n_endpoints = 6
    #n_endpoints = len(endpoints)
    n_runs = 8
    
    main(use_aux_info, use_mbd, n_endpoints, n_runs)