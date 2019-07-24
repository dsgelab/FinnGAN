import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from params import *

def get_survival_analysis_input(data, ENDPOINT, event_name, predictor_name, sequence_length):
    event_i = ENDPOINT.vocab.stoi[event_name]
    predictor_i = ENDPOINT.vocab.stoi[predictor_name]
    
    col_idx = torch.arange(sequence_length)
    
    start_times = []
    end_times = []
    
    predictors = (data == predictor_i).byte()
    events = (data == event_i).byte()
    
    preds = predictors.any(dim = 1)
    outcomes = events.any(dim = 1)
    
    for i in range(data.shape[0]):
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
            
        start_times.append(int(start_time))
        end_times.append(int(end_time))
        
    start_times = pd.Series(start_times, name='start_time')
    end_times = pd.Series(end_times, name='end_time')
        
    preds = pd.Series(preds.numpy(), name='predictor')
    outcomes = pd.Series(outcomes.numpy(), name='outcome')
    
    res = pd.DataFrame({
        'predictor': preds, 
        'outcome': outcomes,
        'duration': end_times - start_times
    })
    
    return res
    


def analyse(event_name, predictor_name):
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = 30_000_000)
    print('Data loaded, number of individuals:', n_individuals)
    
    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    G.load_state_dict(torch.load(G_filename))
    G.eval()
    
    data, data_fake = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length)
    
    
    
    print('REAL:')
    surv_inp = get_survival_analysis_input(data, ENDPOINT, event_name, predictor_name, sequence_length)
    
    cph = CoxPHFitter()
    cph.fit(surv_inp, duration_col='duration', event_col='outcome', show_progress=True)
    cph.print_summary()  # access the results using cph.summary
    print()

    
    
    print('FAKE:')
    surv_inp = get_survival_analysis_input(data_fake, ENDPOINT, event_name, predictor_name, sequence_length)

    cph = CoxPHFitter()
    cph.fit(surv_inp, duration_col='duration', event_col='outcome', show_progress=True)
    cph.print_summary()  # access the results using cph.summary
    print()


if __name__ == '__main__':
    event_name = 'I9_CHD'
    predictor_name = 'C3_BREAST'
    analyse(event_name, predictor_name)