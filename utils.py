from math import log10, floor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchtext
from torchtext.data import Field, Iterator, Dataset, Example
import matplotlib.pyplot as plt

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


def get_sequence_of_codes(subject, sequence_length):
    codes = []
    
    count = 0
    for i in subject.sort_values('EVENT_AGE').index:
        codes.append(subject.loc[i, 'ENDPOINT'])
        count += 1
        if count == sequence_length:
            break
        
    res = ' '.join(codes)
    return res

def get_sequence_of_time_differences(subject, sequence_length):
    times = [0]
    
    count = 0
    for i in subject.sort_values('EVENT_AGE').index:
        times.append(subject.loc[i, 'EVENT_AGE'])
        count += 1
        if count == sequence_length:
            break
        
    res = np.diff(times)
    return res



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
    transition_count = torch.zeros(vocab_size - 1, vocab_size - 1)

    for indv in data:
        for idx in range(len(indv) - d):
            i1 = idx
            i2 = i1 + d
            ep1 = indv[i1]
            ep2 = indv[i2]
            if ep1 > 0 and ep2 > 0:
                transition_count[ep1 - 1, ep2 - 1] += 1
                    
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
        data_tmp = batch.ENDPOINTS.transpose(0, 1)

        start_tokens = data_tmp[:, :1]
        memory = G.initial_state(batch_size = start_tokens.shape[0])

        if cuda:
            start_tokens = start_tokens.cuda()
            memory = memory.cuda()

        _, data_fake_tmp, _, _ = G(start_tokens, memory, sequence_length)
        
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
        data_tmp = batch.ENDPOINTS.transpose(0, 1)
        data.append(data_tmp)

        start_tokens = data_tmp[:, :1]
        memory = G.initial_state(batch_size = start_tokens.shape[0])

        if cuda:
            start_tokens = start_tokens.cuda()
            memory = memory.cuda()

        _, data_fake_tmp, _, _ = G(start_tokens, memory, sequence_length)
        
        data_fake.append(data_fake_tmp.cpu())
    
    
    data = torch.cat(data)
    data_fake = torch.cat(data_fake)
    
    transition_count_real, transition_freq_real = get_transition_matrix(data, vocab_size, d)
    transition_count_fake, transition_freq_fake = get_transition_matrix(data_fake, vocab_size, d)
    
    chi_sqrd_ds = []
    for i in range(vocab_size - 1):
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

