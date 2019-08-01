import pandas as pd
import numpy as np
import torch
from utils import *
from params import *


def save(data, data_fake, train = True):
    df = pd.DataFrame(data.numpy())
    df_fake = pd.DataFrame(data_fake.numpy())
    
    print(df.head())
    print(df_fake.head())
    
    df.to_csv('data/real_{}.csv.gz'.format('train' if train else 'val'))
    df_fake.to_csv('data/fake_{}.csv.gz'.format('train' if train else 'val'))
    
    
if __name__ == '__main__':
    nrows = 3_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    print('Data loaded, number of individuals:', n_individuals)

    G = RelationalMemoryGenerator(mem_slots, head_size, embed_size, vocab_size, temperature, num_heads, num_blocks)
    G.load_state_dict(torch.load(G_filename))
    G.eval()
    
    data, data_fake = get_real_and_fake_data(G, train, ignore_similar, dummy_batch_size, sequence_length, False)
    
    save(data, data_fake)