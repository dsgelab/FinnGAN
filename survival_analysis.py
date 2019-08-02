import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from params import *
from relational_rnn_models import RelationalMemoryGenerator
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt

def get_survival_analysis_input(data, ENDPOINT, event_name, predictor_name):
    event_i = ENDPOINT.vocab.stoi[event_name]
    predictor_i = ENDPOINT.vocab.stoi[predictor_name]
    
    sequence_length = data.shape[1]
    col_idx = torch.arange(sequence_length)
    
    n = data.shape[0]
    start_times = np.zeros((n,))
    end_times = np.zeros((n,))
    
    new_start_times = []
    new_end_times = []
    
    predictors = (data == predictor_i).byte()
    events = (data == event_i).byte()
    
    preds = predictors.any(dim = 1)
    outcomes = events.any(dim = 1)
    
    new_preds = []
    new_outcomes = []
    
    for i in range(n):
        pred = predictors[i, :]
        event = events[i, :]
        
        end_time = sequence_length
        times = col_idx[event]
        if times.shape[0] > 0:
            end_time = times[0]
        
        start_time = 0
        times = col_idx[pred]
        if times.shape[0] > 0:
            start_time = times[0]
            
        if start_time > end_time:
            start_time = 0
            preds[i] = False
            
        if start_time > 0:
            new_start_time = 0
            new_end_time = start_time
            
            new_start_times.append(new_start_time)
            new_end_times.append(new_end_time)
            new_preds.append(False)
            new_outcomes.append(False)
            
        start_times[i] = int(start_time)
        end_times[i] = int(end_time)
        
    start_times = np.concatenate([start_times, new_start_times])
    end_times = np.concatenate([end_times, new_end_times])
    
    start_times = pd.Series(start_times, name='start_time')
    end_times = pd.Series(end_times, name='end_time')
    
    preds = np.concatenate([preds.numpy(), new_preds])
    outcomes = np.concatenate([outcomes.numpy(), new_outcomes])
        
    preds = pd.Series(preds, name='predictor')
    outcomes = pd.Series(outcomes, name='outcome')
    
    res = pd.DataFrame({
        predictor_name: preds, 
        event_name: outcomes,
        'duration': end_times - start_times
    })
    
    return res
    


def analyse(data, data_fake, before, train, ENDPOINT, event_name, predictor_name):
    print()
    print('REAL:')
    surv_inp = get_survival_analysis_input(data, ENDPOINT, event_name, predictor_name)
    
    cph = CoxPHFitter()
    cph.fit(surv_inp, duration_col='duration', event_col=event_name, show_progress=True)
    cph.print_summary()  # access the results using cph.summary
    #cph.check_assumptions(surv_inp, p_value_threshold=0.05, show_plots=False)
    #cph.plot_covariate_groups(predictor_name, [0, 1], plot_baseline=False)
    #plt.title(event_name + ' (real)')
    #plt.savefig('figs/{}_survival_real_{}.svg'.format('Before' if before else 'After', 'train' if train else 'val'))
    
    
    print()
    print('FAKE:')
    surv_inp = get_survival_analysis_input(data_fake, ENDPOINT, event_name, predictor_name)

    cph1 = CoxPHFitter()
    cph1.fit(surv_inp, duration_col='duration', event_col=event_name, show_progress=True)
    cph1.print_summary()  # access the results using cph.summary
    #cph1.check_assumptions(surv_inp, p_value_threshold=0.05, show_plots=False)
    
    cph.plot_covariate_groups(predictor_name, [0, 1], plot_baseline=False)
    cph1.plot_covariate_groups(predictor_name, [0, 1], plot_baseline=False, linestyle='--')
    plt.legend()
    #plt.title(event_name + ' (fake)')
    #plt.savefig('figs/{}_survival_fake_{}.svg'.format('Before' if before else 'After', 'train' if train else 'val'))
    plt.savefig('figs/survival_real_and_fake.jpeg')


if __name__ == '__main__':
    nrows = 9_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    print('Data loaded, number of individuals:', n_individuals)
    
    predictor_name = 'I9_CHD'
    event_name = 'I9_HEARTFAIL_NS'

    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    G.load_state_dict(torch.load(G_filename))
    
    G.eval()
    
    data, data_fake = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length)
    
    analyse(data, data_fake, None, None, ENDPOINT, event_name, predictor_name)