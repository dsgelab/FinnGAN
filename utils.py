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
from params import *
import catheat


cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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
    
    if subject['ENDPOINT'].isin(endpoints).any():
        years = subject.groupby('EVENT_YEAR')
        
        for g, year in years:
            if year['ENDPOINT'].isin(endpoints).any():
                value = 'None'
                if year['ENDPOINT'].isin(['C3_BREAST']).any():
                    value = 'C3_BREAST'
                elif year['ENDPOINT'].isin(['I9_CHD']).any():
                    value = 'I9_CHD'
                else:
                    tmp = year['ENDPOINT'].unique()
                    li = pd.Series(tmp).isin(endpoints)
                    possible_endpoints = tmp[li]
                    if len(possible_endpoints) > 0:
                        value = np.random.choice(possible_endpoints)
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

    
def get_transition_matrix(data, vocab_size, d = 1, ignore_time = False, eps = 1e-20):
    if ignore_time:
        transition_freq = torch.ones(vocab_size - 3, vocab_size - 3) # no <unk>, <pad>, or None
        
        for v1 in range(3, vocab_size):
            for v2 in range(v1, vocab_size):
                li1 = data == v1
                li2 = data == v2
                
                cumsum1 = torch.cumsum(li1, dim = 1)
                cumsum2 = torch.cumsum(li2, dim = 1)
                
                sum1 = torch.sum(cumsum1 == 0, dim = 1)
                sum2 = torch.sum(cumsum2 == 0, dim = 1)
                
                res1 = (sum1 <= sum2) & (sum2 < data.shape[1])
                res2 = (sum2 <= sum1) & (sum1 < data.shape[1])
                
                transition_freq[v1 - 3, v2 - 3] = res1.float().mean()
                transition_freq[v2 - 3, v1 - 3] = res2.float().mean()
            
    else:
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
                    
    return transition_freq





# Define generator evaluation functions

def get_real_and_fake_data(G, dataset, ignore_similar, batch_size, sequence_length, include_age_and_sex = False):
    iterator = Iterator(dataset, batch_size = batch_size)
    
    if cuda:
        G.cuda()
    
    data = []
    data_fake = []
    
    if include_age_and_sex:
        ages = []
        sexes = []
    
    for batch in iterator:
        data_tmp = batch.ENDPOINT.transpose(0, 1)
        data.append(data_tmp)
        
        if include_age_and_sex:
            ages.append(batch.AGE)
            sexes.append(batch.SEX.view(-1))

        start_tokens = data_tmp[:, :1]

        if cuda:
            start_tokens = start_tokens.cuda()

        _, data_fake_tmp, _ = G(start_tokens, batch.AGE, batch.SEX.view(-1), None, sequence_length)
        
        data_fake.append(data_fake_tmp.cpu())
    
    data = torch.cat(data)
    data_fake = torch.cat(data_fake)
    
    data = data[:, 1:]
    data_fake = data_fake[:, 1:]
    
    if include_age_and_sex:
        ages = torch.cat(ages)
        sexes = torch.cat(sexes)
        
        ages += 1
    
    # Filter those fake samples out which have at least 1 exact match in the real data
    if ignore_similar:
        li = robust_get_similarity_score(data_fake, data, dummy_batch_size2, True)
        data_fake = data_fake[~li, :]
        
        if include_age_and_sex:
            ages_fake = ages[~li]
            sexes_fake = sexes[~li]
        
            return data, ages, sexes, data_fake, ages_fake, sexes_fake
        
    if include_age_and_sex:
        return data, ages, sexes, data_fake

    return data, data_fake

# More interpretable version of chi_sqrd_dist
def get_diffs(dist, target, separate = False):
    abs_diffs = torch.abs(dist - target)
    
    #max_abs_diffs, _ = torch.stack([torch.ones(target.shape) - target, target], dim = 1).max(dim = 1)
    
    relative_diffs = abs_diffs #/ max_abs_diffs
    
    if separate:
        return relative_diffs
    
    return relative_diffs.mean()

    
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
    
def get_score(data_fake, ENDPOINT, vocab_size):
    counts_real, freqs_real = get_distribution(None, ENDPOINT, vocab_size, fake = False)
    
    counts_fake, freqs_fake = get_distribution(data_fake, None, vocab_size, fake = True)
    
    #score = chi_sqrd_dist(counts_fake, counts_real)
    score = get_diffs(freqs_fake, freqs_real)
    return score

def get_transition_score(data, data_fake, d, ignore_time, separate, vocab_size):
    transition_freq_real = get_transition_matrix(data, vocab_size, d, ignore_time)
    transition_freq_fake = get_transition_matrix(data_fake, vocab_size, d, ignore_time)
    
    if ignore_time:
        res = (transition_freq_real - transition_freq_fake).abs()
                
        if separate:
            return res
        
        return torch.mean(res, dim = 1)
        
    else:
        chi_sqrd_ds = []
        for i in range(transition_freq_real.shape[0]):
            #chi_sqrd_d = chi_sqrd_dist(transition_count_fake[i, :], transition_count_real[i, :])
            chi_sqrd_d = get_diffs(transition_freq_fake[i, :], transition_freq_real[i, :])
            chi_sqrd_ds.append(chi_sqrd_d)

        chi_sqrd_ds = torch.tensor(chi_sqrd_ds)

        if separate:
            return chi_sqrd_ds

        return torch.mean(chi_sqrd_ds)
    
def get_aggregate_transition_score(data, data_fake, ignore_time, separate1, separate2, vocab_size, sequence_length):
    if ignore_time:
        result = get_transition_score(data, data_fake, None, True, separate1, vocab_size)
    else:
        scores = []
        for d in range(1, sequence_length):
            transition_score = get_transition_score(data, data_fake, d, False, separate1, vocab_size)
            scores.append(transition_score)

        result = torch.stack(scores)

    if separate2:
        return result

    if separate1:
        return torch.mean(result, dim = 0)
    else:
        return torch.mean(result)
    
def get_similarity_score(data1, data2, separate):
    n = data1.shape[0]
    m = data2.shape[0]
    res = torch.zeros(n, m)
    
    for i in range(m):
        res[:, i] = (data1 == data2[i]).all(dim = 1)
        
    res = res.byte().any(dim = 1)
    
    if separate:
        return res
            
    return res.float().mean()

def robust_get_similarity_score(data1, data2, batch_size, separate):
    lis = []
    
    data2 = data2.unique(dim = 0)
        
    for i in range(0, data1.shape[0], batch_size):
        data1_tmp = data1[i:i+batch_size, :]
        li = get_similarity_score(data1_tmp, data2, True)
        lis.append(li)
        
    res = torch.cat(lis)
    
    if separate:
        return res
            
    return res.float().mean()
    
    
def get_individual_distribution(data, vocab_size, sequence_length):
    individual_counts = torch.zeros(vocab_size - 3, sequence_length + 1)
    
    for v in range(3, vocab_size):
        counts = torch.sum(data == v, dim = 1)
        
        for i in range(sequence_length + 1):
            individual_counts[v - 3, i] += torch.sum(counts == i)
            
    individual_freqs = (individual_counts.transpose(0, 1) / torch.sum(individual_counts, dim = 1)).transpose(0, 1)
    
    return individual_counts, individual_freqs


def get_individual_score(data, data_fake, separate, vocab_size, sequence_length):
    individual_counts_real, individual_freqs_real = get_individual_distribution(data, vocab_size, sequence_length)
    
    individual_counts_fake, individual_freqs_fake = get_individual_distribution(data_fake, vocab_size, sequence_length)
    
    individual_scores = torch.zeros(individual_counts_real.shape[0])
    
    for i in range(individual_counts_real.shape[0]):
        #individual_scores[i] = chi_sqrd_dist(individual_counts_fake[i, :], individual_counts_real[i, :])
        individual_scores[i] = get_diffs(individual_freqs_fake[i, :], individual_freqs_real[i, :])
        
    if separate:
        return individual_scores
    
    return individual_scores.mean()




def save_grouped_barplot(freqs, freqs_fake, idx, field, title, N=10):
    plt.style.use('classic')
    plt.style.use(plot_style)
    
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
        
        

def get_scores(G, ENDPOINT, dataset, batch_size, ignore_time, separate1, separate2, ignore_similar, vocab_size, sequence_length):
    
    G.eval()
    
    data, data_fake = get_real_and_fake_data(G, dataset, ignore_similar, batch_size, sequence_length)
    
    if ignore_similar:
        similarity_score = torch.tensor(1.0 - data_fake.shape[0] / data.shape[0])
    else:
        similarity_score = robust_get_similarity_score(data, data_fake, dummy_batch_size2, False)
        

    score1 = get_score(data_fake, ENDPOINT, vocab_size)
    
    transition_score = get_aggregate_transition_score(data, data_fake, ignore_time, separate1, separate2, vocab_size, sequence_length)
    
    indv_score = get_individual_score(data, data_fake, separate1, vocab_size, sequence_length)
    
    G.train()
    
    return score1, transition_score.mean(), similarity_score, indv_score.mean(), transition_score, indv_score


def get_dataset(nrows = 3_000_000):
    filename = 'data/FINNGEN_ENDPOINTS_DF3_longitudinal_V1_for_SandBox.txt.gz'

    events = pd.read_csv(filename, compression = 'infer', sep='\t', nrows = nrows)
    
    subjects = events['FINNGENID'].unique()
    n_individuals = len(subjects)

    # include all endpoints in a list
    #events = events[events['ENDPOINT'].isin(endpoints)] # comment/uncomment this to include/exclude emtpy subjects
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

    # TODO: split data first into a train and test set; use the same random state
    train_sequences, val_sequences = train_test_split(sequences, test_size = 0.1)

    train = DataFrameDataset(train_sequences, fields)
    val = DataFrameDataset(val_sequences, fields)

    ENDPOINT.build_vocab(train, val)
    SEX.build_vocab(train, val)

    vocab_size = len(ENDPOINT.vocab.freqs) + 2
    
    return train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals

def save_plots_of_train_scores(scores1_train, transition_scores_mean_train, similarity_score_train, indv_score_mean_train, transition_scores_train, indv_score_train, \
    scores1_val, transition_scores_mean_val, similarity_score_val, indv_score_mean_val, transition_scores_val, indv_score_val, \
    accuracies_real, accuracies_fake, ignore_time, sequence_length, vocab_size, ENDPOINT):
    x = np.arange(scores1_train.shape[0]) - 2
    
    plt.style.use(plot_style)
    
    y = scores1_train.numpy()
    plt.plot(x, y, label='training set')
    y = scores1_val.numpy()
    plt.plot(x, y, label='validation set')
    plt.ylim(0, 1)
        
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Distribution score')
    plt.xlabel('Iteration')
    plt.savefig('figs/chisqrd_freqs.svg')
    plt.clf()

    
    y = transition_scores_mean_train.numpy()
    plt.plot(x, y, label='training set')
    y = transition_scores_mean_val.numpy()
    plt.plot(x, y, label='validation set')
    plt.ylim(0, 1)
        
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Mean transition score')
    plt.xlabel('Iteration')
    plt.savefig('figs/mean_transition_score.svg')
    plt.clf()

    
    y = similarity_score_train.numpy()
    plt.plot(x, y, label='training set')
    y = similarity_score_val.numpy()
    plt.plot(x, y, label='validation set')
    plt.ylim(0, 1)
    
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Similarity score')
    plt.xlabel('Iteration')
    plt.savefig('figs/similarity_score.svg')
    plt.clf()
    
    
    y = indv_score_mean_train.numpy()
    plt.plot(x, y, label='training set')
    y = indv_score_mean_val.numpy()
    plt.plot(x, y, label='validation set')
    plt.ylim(0, 1)
        
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Mean individual score')
    plt.xlabel('Iteration')
    plt.savefig('figs/mean_indv_score.svg')
    plt.clf()

    
    y = accuracies_real.detach().cpu().numpy()
    plt.plot(range(y.shape[0]), y)
    plt.ylabel('Accuracy real')
    plt.xlabel('Iteration')
    plt.savefig('figs/accuracy_real.svg')
    plt.clf()

    
    y = accuracies_fake.detach().cpu().numpy()
    plt.plot(range(y.shape[0]), y)
    plt.ylabel('Accuracy fake')
    plt.xlabel('Iteration')
    plt.savefig('figs/accuracy_fake.svg')
    plt.clf()
    
    
    y = indv_score_val.numpy()
    plt.plot(x, y)
    plt.ylim(0, 1)
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.ylabel('Individual score')
    plt.xlabel('Iteration')
    labels = ['endpoint=' + ENDPOINT.vocab.itos[i] for i in range(3, vocab_size)]
    plt.legend(labels)
    plt.savefig('figs/indv_score.svg')
    plt.clf()

    
    if ignore_time:
        for v in range(3, vocab_size):
            y = transition_scores_val[:, v - 3, :].numpy()
            plt.plot(x, y)
            plt.ylim(0, 1)
            plt.axvline(-1, color='k', linestyle='--', label='pretraining')
            plt.ylabel('Transition score')
            plt.xlabel('Iteration')
            title = 'enpoint=' + ENDPOINT.vocab.itos[v]
            plt.title(title)
            labels = ['endpoint=' + ENDPOINT.vocab.itos[i] for i in range(3, vocab_size)]
            plt.legend(labels)
            plt.savefig('figs/' + title + '.svg')
            plt.clf()
    else:
        for v in range(2, vocab_size):
            y = scores3_val[:, :, v - 2].numpy()
            plt.plot(x, y)
            plt.ylim(0, 1)
            plt.axvline(-1, color='k', linestyle='--', label='pretraining')
            plt.ylabel('Transition score')
            plt.xlabel('Iteration')
            title = 'enpoint=' + ENDPOINT.vocab.itos[v]
            plt.title(title)
            labels = ['d=' + str(i) for i in range(1, sequence_length)]
            plt.legend(labels)
            plt.savefig('figs/' + title + '.svg')
            plt.clf()
            
            
    y = scores1_train.numpy()
    plt.plot(x, y, label='distribution score')
    y = transition_scores_mean_train.numpy()
    plt.plot(x, y, label='transition score')
    y = similarity_score_train.numpy()
    plt.plot(x, y, label='similarity score')
    plt.ylim(0, 1)
    
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Value')
    plt.xlabel('Iteration')
    plt.savefig('figs/combined_train.svg')
    plt.clf()
    
            
    y = scores1_val.numpy()
    plt.plot(x, y, label='distribution score')
    y = transition_scores_mean_val.numpy()
    plt.plot(x, y, label='transition score')
    y = similarity_score_val.numpy()
    plt.plot(x, y, label='similarity score')
    plt.ylim(0, 1)
    
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.ylabel('Value')
    plt.xlabel('Iteration')
    plt.savefig('figs/combined_val.svg')
    plt.clf()
    
    
    
def plot_data(data, ages, sexes, ENDPOINT, SEX, N=10, save=True, filename='figs/catheat.svg'):
    plt.style.use(plot_style)
    
    data = data[:N, :].cpu().numpy()
    
    new_data = np.empty(data.shape, dtype = 'object')
    for row, col in itertools.product(range(data.shape[0]), range(data.shape[1])):
        new_data[row, col] = ENDPOINT.vocab.itos[data[row, col]]
    
    cmap = {
        'None': '#FFFFFF',
        'I9_HYPTENS': '#00FFDB', 
        'I9_ANGINA': '#00AF66', 
        'I9_HEARTFAIL_NS': '#FF88FF', 
        'I9_STR_EXH': '#8D88D6', 
        'I9_CHD': '#0073D6', 
        'C3_BREAST': '#BE1200',
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
    
    # TODO: don't output empty subjects

    if cuda:
        start_tokens = start_tokens.cuda()

    _, data_fake, _ = G(start_tokens, batch.AGE, batch.SEX.view(-1), None, sequence_length)
    
    plot_data(data_real, batch.AGE.view(-1), batch.SEX.view(-1), ENDPOINT, SEX, N=size, save=True, filename='figs/catheat_real.svg')

    plot_data(data_fake, batch.AGE.view(-1), batch.SEX.view(-1), ENDPOINT, SEX, N=size, save=True, filename='figs/catheat_fake.svg')
    
def save_frequency_comparisons(data_fake1, data_fake2, vocab_size, ENDPOINT, prefix, N_max):
    counts_fake1, _ = get_distribution(data_fake1, None, vocab_size, fake = True)
    counts_fake2, _ = get_distribution(data_fake2, None, vocab_size, fake = True)

    counts_fake = counts_fake1 + counts_fake2
    freqs_fake = counts_fake / torch.sum(counts_fake)

    counts, freqs = get_distribution(None, ENDPOINT, vocab_size, fake = False)

    save_relative_and_absolute(freqs, freqs_fake, counts, counts_fake, vocab_size, ENDPOINT, prefix, N_max)

