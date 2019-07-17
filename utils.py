from math import log10, floor
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
import itertools
from torchtext.data import Field, Iterator, Dataset, Example
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from params import *
import catheat


cuda = torch.cuda.is_available()


# Helper function(s)

# round a number to n significant digits
def round_to_n(x, n = 2):
    return round(x, -int(floor(log10(abs(x)))) + (n - 1)) if x != 0 else 0

# Transform a date string into a datetime object
def str_to_datetime(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

# TODO: more complex reduction: takes into account different ICD groups?
def reduce_icd(icd_full):
    return icd_full[:2]

def get_distribution(data, field, vocab_size, fake = True):
    counts = torch.zeros(vocab_size - 2)

    for i in range(2, vocab_size):
        if fake:
            counts[i - 2] = torch.sum(data == i)
        else:
            counts[i - 2] = field.vocab.freqs[field.vocab.itos[i]]

    freqs = counts / torch.sum(counts)
    
    return counts, freqs


def get_sequence_of_codes(subject):
    codes = ['None' for _ in range(2017 - 2000 + 1)]
    
    years = subject.groupby('EVENT_YEAR')
    
    for g, year in years:
        if year['ENDPOINT'].isin(['C3_BREAST']).any():
            value = 'C3_BREAST'
        elif year['ENDPOINT'].isin(['I9_CHD']).any():
            value = 'I9_CHD'
        else:
            value = np.random.choice(year['ENDPOINT'])
        codes[g - 2000] = value
        
    res = ' '.join(codes)
    return res

def get_age(subject):
    event = subject.sort_values('EVENT_AGE').iloc[0]
    
    age = event['EVENT_AGE'] + 2000 - event['EVENT_YEAR']
        
    return age



# https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext

class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.fields = dict(fields)
        self.examples = examples.apply(SeriesExample.fromSeries, args=(self.fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""
    
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex

    
def get_transition_matrix(data, vocab_size, d = 1, eps = 1e-20):
    transition_count = torch.zeros(vocab_size - 2, vocab_size - 2)

    for indv in data:
        for idx in range(len(indv) - d):
            i1 = idx
            i2 = i1 + d
            ep1 = indv[i1]
            ep2 = indv[i2]
            if ep1 > 1 and ep2 > 1:
                transition_count[ep1 - 2, ep2 - 2] += 1
                    
    #print(torch.sum(transition_count, dim = 1))
    transition_freq = (transition_count.transpose(0, 1) / (torch.sum(transition_count, dim = 1) + eps)).transpose(0, 1)
                    
    return transition_count, transition_freq


# Define generator evaluation functions

def chi_sqrd_dist(counts1, counts2, separate = False, eps = 1e-20):
    counts1 = counts1.view(1, -1)
    counts2 = counts2.view(1, -1)
    table = torch.cat([counts1, counts2], dim = 0)
    col_sums = torch.sum(table, dim = 0)
    row_sums = torch.sum(table, dim = 1)
    n = torch.sum(col_sums)
    
    table_freq = table / (n + eps)
    col_freqs = col_sums / (n + eps)
    row_freqs = row_sums / (n + eps)
    
    diffs = table_freq[0, :] / (row_freqs[0] + eps) - table_freq[1, :] / (row_freqs[1] + eps)
    diffs_sqrd = diffs ** 2
    diffs_sqrd_norm = diffs_sqrd / (col_freqs + eps)
    
    if separate: 
        return diffs_sqrd_norm
    
    chi_sqrd_distance = torch.sum(diffs_sqrd_norm)
    
    return chi_sqrd_distance

def get_fake_distribution(G, dataset, batch_size, vocab_size, sequence_length):
    iterator = Iterator(dataset, batch_size = batch_size)
    
    if cuda:
        G.cuda()
    
    data_fake = []
    
    for batch in iterator:
        data_tmp = batch.ENDPOINT.transpose(0, 1)

        start_tokens = data_tmp[:, :1]
        memory = G.initial_state(batch_size = start_tokens.shape[0])

        if cuda:
            start_tokens = start_tokens.cuda()
            memory = memory.cuda()

        _, data_fake_tmp, _, _ = G(start_tokens, batch.AGE, batch.SEX, memory, sequence_length)
        
        data_fake.append(data_fake_tmp.cpu())
    
    data_fake = torch.cat(data_fake)
    
    counts_fake, freqs_fake = get_distribution(data_fake, None, vocab_size, fake = True)
    
    return counts_fake, freqs_fake
    
def get_score(G, ENDPOINT, dataset, batch_size, vocab_size, sequence_length):
    counts_real, freqs_real = get_distribution(None, ENDPOINT, vocab_size, fake = False)
    
    counts_fake, freqs_fake = get_fake_distribution(G, dataset, batch_size, vocab_size, sequence_length)
    
    score = chi_sqrd_dist(counts_real, counts_fake)
    return score

def get_transition_score(G, dataset, batch_size, d, separate, vocab_size, sequence_length):
    iterator = Iterator(dataset, batch_size = batch_size)
    
    if cuda:
        G.cuda()
    
    data = []
    data_fake = []
    
    for batch in iterator:
        data_tmp = batch.ENDPOINT.transpose(0, 1)
        data.append(data_tmp)

        start_tokens = data_tmp[:, :1]
        memory = G.initial_state(batch_size = start_tokens.shape[0])

        if cuda:
            start_tokens = start_tokens.cuda()
            memory = memory.cuda()

        _, data_fake_tmp, _, _ = G(start_tokens, batch.AGE, batch.SEX, memory, sequence_length)
        
        data_fake.append(data_fake_tmp.cpu())
    
    
    data = torch.cat(data)
    data_fake = torch.cat(data_fake)
    
    transition_count_real, transition_freq_real = get_transition_matrix(data, vocab_size, d)
    transition_count_fake, transition_freq_fake = get_transition_matrix(data_fake, vocab_size, d)
    
    chi_sqrd_ds = []
    for i in range(transition_count_real.shape[0]):
        chi_sqrd_d = chi_sqrd_dist(transition_count_fake[i, :], transition_count_real[i, :])
        chi_sqrd_ds.append(chi_sqrd_d)
        
    chi_sqrd_ds = torch.tensor(chi_sqrd_ds)
    
    if separate:
        return chi_sqrd_ds
        
    return torch.mean(chi_sqrd_ds)
    
def get_aggregate_transition_score(G, dataset, batch_size, separate1, separate2, vocab_size, sequence_length):
    scores = []
    for d in range(1, sequence_length):
        transition_score = get_transition_score(G, dataset, batch_size, d, separate1, vocab_size, sequence_length)
        scores.append(transition_score)
        
    result = torch.stack(scores)
    
    if separate2:
        return result
    
    if separate1:
        return torch.mean(result, dim = 0)
    else:
        return torch.mean(result)



def save_grouped_barplot(freqs, freqs_fake, idx, field, title, N=10):
    freqs1 = freqs.numpy()[idx]
    freqs2 = freqs_fake.numpy()[idx]

    fig, ax = plt.subplots(figsize=(6,6))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    p1 = ax.bar(ind, freqs1, width, bottom=0)

    p2 = ax.bar(ind + width, freqs2, width, bottom=0)

    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(map(lambda x: field.vocab.itos[x], idx + 2), rotation=90)

    ax.legend((p1[0], p2[0]), ('real', 'fake'))
    ax.autoscale_view()
    
    fig.subplots_adjust(bottom=0.36)
    fig.savefig('figs/' + '_'.join(title.split(' ')).translate({ord(i): None for i in ':()'}) + '.svg')
    plt.clf()
    
def save_relative_and_absolute(freqs, freqs_fake, counts, counts_fake, vocab_size, field, prefix='', N_max=10):
    N = min(N_max, vocab_size - 2)
    if not prefix == '':
        prefix += ' '
    
    if N == vocab_size - 2:
        idx = np.arange(N)
        
        title = prefix + 'Differences in frequencies'
        save_grouped_barplot(freqs, freqs_fake, idx, field, title, N)
    else:
        abs_diffs = np.abs(freqs - freqs_fake)
        idx = np.argsort(abs_diffs)[-N:]
        idx = np.flip(idx)
        
        title = prefix + 'Largest absolute differences in frequencies'
        save_grouped_barplot(freqs, freqs_fake, idx, field, title, N)

        chi_sqrd_dists = chi_sqrd_dist(counts, counts_fake, separate = True)
        idx = np.argsort(chi_sqrd_dists)[-N:]
        idx = np.flip(idx)

        title = prefix + 'Largest relative differences in frequencies'
        save_grouped_barplot(freqs, freqs_fake, idx, field, title, N)

# Define the training function

def get_scores(G, ENDPOINT, dataset, batch_size, separate1, separate2, vocab_size, sequence_length):
    score1 = get_score(G, ENDPOINT, dataset, batch_size, vocab_size, sequence_length)
    
    score2 = get_aggregate_transition_score(G, dataset, batch_size, separate1, separate2, vocab_size, sequence_length)
    
    return score1, score2.mean(), score2

def get_dataset(nrows = 3_000_000):
    filename = 'data/FINNGEN_ENDPOINTS_DF3_longitudinal_V1_for_SandBox.txt.gz'

    endpoints = ['I9_HYPTENS', 'I9_ANGINA', 'I9_HEARTFAIL_NS', 'I9_STR_EXH', 'I9_CHD', 'C3_BREAST']

    events = pd.read_csv(filename, compression = 'infer', sep='\t', nrows = nrows)

    # include all endpoints in a list
    events = events[events['ENDPOINT'].isin(endpoints)]
    #events = events.groupby('FINNGENID').filter(lambda x: len(x) > 1)
    events = events[events['EVENT_YEAR'] >= 2000]
    
    filename = 'data/FINNGEN_MINIMUM_DATA_R3_V1.txt'

    patients = pd.read_csv(filename, compression = 'infer', sep='\t')
    events = pd.merge(events, patients[['FINNGENID', 'SEX']])

    subjects = events['FINNGENID'].unique()
    n_individuals = len(subjects)

    #sequence_length = min(events.groupby('FINNGENID').apply(lambda x: len(x)).max(), max_sequence_length)
    sequence_length = 2017 - 2000 + 1

    sequences_of_codes = events.groupby('FINNGENID').apply(get_sequence_of_codes)
    ages = events.groupby('FINNGENID').apply(get_age)
    sexes = events.groupby('FINNGENID')['SEX'].first()

    sequences = pd.DataFrame({'ENDPOINT': sequences_of_codes, 'AGE': ages, 'SEX': sexes})


    tokenize = lambda x: x.split(' ')
    
    ENDPOINT = Field(tokenize = tokenize)
    AGE = Field(sequential = False, use_vocab = False)
    SEX = Field()

    fields = [('ENDPOINT', ENDPOINT), ('AGE', AGE), ('SEX', SEX)]

    train_sequences, val_sequences = train_test_split(sequences, test_size = 0.1)

    train = DataFrameDataset(train_sequences, fields)
    val = DataFrameDataset(val_sequences, fields)

    ENDPOINT.build_vocab(train, val)
    SEX.build_vocab(train, val)

    vocab_size = len(ENDPOINT.vocab.freqs) + 2
    
    return train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals

def save_plots_of_train_scores(scores1, scores2, scores3, accuracies_real, accuracies_fake, sequence_length, vocab_size, ENDPOINT):
    plt.plot(range(scores1.shape[0]), scores1.numpy())
    plt.ylabel('Chi-Squared Distance of frequencies')
    plt.xlabel('Epoch')
    plt.savefig('figs/chisqrd_freqs.svg')
    plt.clf()

    plt.plot(range(scores2.shape[0]), scores2.numpy())
    plt.ylabel('Mean transition score')
    plt.xlabel('Epoch')
    plt.savefig('figs/mean_transition_score.svg')
    plt.clf()


    plt.plot(range(accuracies_real.shape[0]), accuracies_real.detach().cpu().numpy())
    plt.ylabel('Accuracy real')
    plt.xlabel('Epoch')
    plt.savefig('figs/accuracy_real.svg')
    plt.clf()


    plt.plot(range(accuracies_fake.shape[0]), accuracies_fake.detach().cpu().numpy())
    plt.ylabel('Accuracy fake')
    plt.xlabel('Epoch')
    plt.savefig('figs/accuracy_fake.svg')
    plt.clf()

    '''
    for d in range(1, sequence_length):
        plt.plot(range(scores3.shape[0]), scores3[:, d - 1, :].numpy())
        plt.ylabel('Transition score')
        plt.xlabel('Epoch')
        title = 'd=' + str(d)
        plt.title(title)
        labels = [ENDPOINT.vocab.itos[i] for i in range(1, vocab_size)]
        plt.legend(labels)
        plt.savefig('figs/' + title + '.svg')
        plt.clf()
    '''

    for v in range(2, vocab_size):
        plt.plot(range(scores3.shape[0]), scores3[:, :, v - 2].numpy())
        plt.ylabel('Transition score')
        plt.xlabel('Epoch')
        title = 'enpoint=' + ENDPOINT.vocab.itos[v]
        plt.title(title)
        labels = ['d=' + str(i) for i in range(1, sequence_length)]
        plt.legend(labels)
        plt.savefig('figs/' + title + '.svg')
        plt.clf()
    
    
    
def plot_data(data, ages, sexes, ENDPOINT, SEX, N=10, save=True, filename='figs/catheat.svg'):
    data = data[:N, :].cpu().numpy()
    
    new_data = np.empty(data.shape, dtype = 'object')
    for row, col in itertools.product(range(data.shape[0]), range(data.shape[1])):
        new_data[row, col] = ENDPOINT.vocab.itos[data[row, col]]
    
    cmap = {
        'None': '#ffffff',
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = catheat.heatmap(new_data, cmap = cmap, ax = ax, linewidths = .5, leg_pos = 'top')
    
    labels = list(map(lambda x: SEX.vocab.itos[sexes[x]] + ', ' + str(int(ages[x])), range(N)))
    plt.yticks(np.arange(N) + 0.5, labels, rotation = 0)
    
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    
def visualize_output(G, size, dataset, sequence_length, ENDPOINT, SEX):
    iterator = Iterator(dataset, size)
    batch = next(iter(iterator))
    data_real = batch.ENDPOINT.transpose(0, 1)
    start_tokens = data_real[:, :1]
    
    plot_data(data_real, batch.AGE.view(-1), batch.SEX.view(-1), ENDPOINT, SEX, N=size, save=True, filename='figs/catheat_real.svg')

    memory = G.initial_state(batch_size = size)

    if cuda:
        memory = memory.cuda()
        start_tokens = start_tokens.cuda()

    _, data_fake, _, _ = G(start_tokens, batch.AGE, batch.SEX, memory, sequence_length)

    plot_data(data_fake, batch.AGE.view(-1), batch.SEX.view(-1), ENDPOINT, SEX, N=size, save=True, filename='figs/catheat_fake.svg')
    
def save_frequency_comparisons(G, train, val, dummy_batch_size, vocab_size, sequence_length, ENDPOINT, prefix, N_max):
    
    counts_fake1, _ = get_fake_distribution(G, val, dummy_batch_size, vocab_size, sequence_length)
    counts_fake2, _ = get_fake_distribution(G, train, dummy_batch_size, vocab_size, sequence_length)

    counts_fake = counts_fake1 + counts_fake2
    freqs_fake = counts_fake / torch.sum(counts_fake)

    counts, freqs = get_distribution(None, ENDPOINT, vocab_size, fake = False)

    save_relative_and_absolute(freqs, freqs_fake, counts, counts_fake, vocab_size, ENDPOINT, prefix, N_max)
