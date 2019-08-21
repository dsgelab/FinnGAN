import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from params import *
import glob
import torch
from collections import OrderedDict
from survival_analysis import plot_survival_functions
import os

def get_true_count(args):
    true_count = 0

    for k, v in args.items():
        if k != 'n_endpoints':
            if v:
                true_count += 1

    return true_count


def smart_label(args):
    true_count = get_true_count(args)
    false_count = len(args) - 1 - true_count

    name_map = {
        'use_aux_info': 'AUX',
        'use_mbd': 'MBD',
        'use_gp': '0-GP',
        'feature_matching': 'FM',
    }

    if true_count == 1:
        for k, v in args.items():
            if k != 'n_endpoints' and v:
                return name_map[k]
    elif true_count == 0:
        return 'BASE'

    strs = []

    for k, v in args.items():
        if k != 'n_endpoints' and v:
            strs.append(k + '=True')

    return ','.join(strs)

def get_args(dirname):
    dirs = dirname.split('/')
    parts = dirs[1].split(' ')

    res = {}

    for p in parts:
        k, v = tuple(p.split('='))
        if k == 'n_endpoints':
            res[k] = int(v)
        else:
            res[k] = v == 'True'
            #print(v, res[k])

    if 'use_gp' not in res:
        res['use_gp'] = False

    if 'feature_matching' not in res:
        res['feature_matching'] = False

    return res

def data_dict_key(item):
    return np.median(item)

def plot_boxes(filename):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    data_dict = {}

    for dirname in dirnames:
        args = get_args(dirname)

        subdirnames = glob.glob(dirname + '/*/')

        tmp_data = []

        for seed_dir in subdirnames:
            tmp = torch.load(seed_dir + filename).numpy()
            tmp_data.append(tmp)

        tmp_data = np.stack(tmp_data, axis = 1)

        if get_true_count(args) <= 1:
            data_dict[smart_label(args)] = tmp_data[-1, :]

    data_dict['MLE'] = tmp_data[1, :] # MLE/pretraining

    #data_dict = sorted(data_dict.items(), key=lambda kv: np.median(kv[1]))
    #data_dict = OrderedDict(data_dict)

    color_list = ['#CF9821', '#98CF21', '#21CF98', '#CF2198', '#2198CF', '#9821CF']
    keys = ['BASE', 'MLE', 'MBD', 'FM', '0-GP', 'AUX']
    ylabels = {
        'chi-sqrd': 'Chi-squared metric',
        'transition': 'Transition metric',
        'similarity1': 'Similairty metric (all sequences)',
        'similarity2': 'Similairty metric (unique sequences)',
        'mode_collapse': 'Mode collapse metric',
    }

    pos = np.arange(1, len(data_dict) + 1)
    for p in pos:
        #parts = plt.violinplot(list(data_dict.values())[p - 1], [p], points=600, widths=0.7,
        #                showextrema=False, showmedians=False)
        #for pc in parts['bodies']:
        #    pc.set_alpha(1)
        plt.boxplot(
            data_dict[keys[p - 1]],
            positions = [p],
            widths=0.4,
            patch_artist=True,
            boxprops = dict(facecolor=color_list[p - 1], alpha=1),
            medianprops = dict(color='k')
        )


    plt.xticks(pos, keys)
    title = filename.split('_val.pt')[0]
    #plt.title(title)
    #plt.show()
    plt.ylabel(ylabels[title])
    plt.savefig('figs/' + title + '.png')
    plt.clf()


def plot_hr(predictor_name, event_name):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    clean_names = {
        'I9_HEARTFAIL_NS': 'heart failure',
        'I9_HYPTENS': 'hypertension',
        'I9_STR_EXH': 'stroke',
        'C3_BREAST': 'breast cancer',
        'I9_CHD': 'CHD',
        'I9_ANGINA': 'angina'
    }

    for dirname in dirnames:
        args = get_args(dirname)

        subdirnames = glob.glob(dirname + '/*/')

        real_hrs = []
        fake_hrs = []

        for seed_dir in subdirnames:
            filename = seed_dir + predictor_name + '->' + event_name + '_hr_real_val.csv'
            if os.path.exists(filename):
                df = pd.read_csv(
                    filename,
                    header = None,
                    index_col = 0
                )
                real_hrs.append(df.values[0])
            filename = seed_dir + predictor_name + '->' + event_name + '_hr_fake_val.csv'
            if os.path.exists(filename):
                df = pd.read_csv(
                    filename,
                    header = None,
                    index_col = 0
                )
                fake_hrs.append(df.values[0])

        real_hrs = np.concatenate(real_hrs)
        fake_hrs = np.concatenate(fake_hrs)

        color_list = ['#CF9821', '#98CF21', '#21CF98', '#CF2198', '#2198CF', '#9821CF']

        data = [real_hrs, fake_hrs]

        pos = np.arange(1, len(data) + 1)
        for p in pos:
            plt.boxplot(
                data[p - 1],
                positions = [p],
                widths=0.4,
                patch_artist=True,
                boxprops = dict(facecolor=color_list[p - 1], alpha=1),
                medianprops = dict(color='k')
            )


        plt.xticks(pos, ['real', 'generated'])
        title = smart_label(args) + '_' +  predictor_name + '->' + event_name + '_hr'
        #plt.title(title)
        #plt.show()
        plt.ylabel('Hazard ratio')
        if get_true_count(args) <= 1:
            plt.savefig('figs/' + title + '.png')
        plt.clf()



def plot_survival(predictor_name, event_name):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    clean_names = {
        'I9_HEARTFAIL_NS': 'heart failure',
        'I9_HYPTENS': 'hypertension',
        'I9_STR_EXH': 'stroke',
        'C3_BREAST': 'breast cancer',
        'I9_CHD': 'CHD',
        'I9_ANGINA': 'angina'
    }

    for dirname in dirnames:
        args = get_args(dirname)

        subdirnames = glob.glob(dirname + '/*/')

        dfs = []
        flag = True

        for seed_dir in subdirnames:
            filename = seed_dir + predictor_name + '->' + event_name + '_val.csv'
            if os.path.exists(filename):
                df = pd.read_csv(
                    filename,
                    index_col = 0
                )
                df = df.fillna(method='ffill')
                dfs.append(df if flag else df.iloc[:, 2:])
                if flag:
                    flag = False
                #plot_survival_functions(df, clean_names, event_name, predictor_name, False, False)

        df = pd.concat(dfs, axis = 1)

        plt.plot(df.iloc[:, 0], linestyle='-', color='b')
        plt.plot(df.iloc[:, 1], linestyle='-', color='g')
        plt.plot(df.iloc[:, 2], linestyle='--', color='b', alpha=0.5)
        plt.plot(df.iloc[:, 3], linestyle='--', color='g', alpha=0.5)
        if len(df.columns) > 4:
            plt.plot(df.iloc[:, 4::2], linestyle='--', color='b', alpha=0.5)
            plt.plot(df.iloc[:, 5::2], linestyle='--', color='g', alpha=0.5)
        plt.legend(df.columns[:4])

        plt.ylabel('Survival probability of developing {}'.format(clean_names[event_name] if event_name in clean_names else event_name))
        plt.xlabel('Time in years')

        if get_true_count(args) <= 1:
            plt.savefig('figs/{}_survival_{}->{}.png'.format(smart_label(args), predictor_name, event_name))
        plt.clf()



# TODO: change this to something reasonnable
def main(filename):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    for dirname in dirnames:
        args = get_args(dirname)
        print(args)

        subdirnames = glob.glob(dirname + '/*/')

        chi_sqrds = []

        for seed_dir in subdirnames:
            chi_sqrd = torch.load(seed_dir + filename).numpy()
            chi_sqrds.append(chi_sqrd)

        chi_sqrds = np.stack(chi_sqrds, axis = 1)

        x = np.arange(chi_sqrds.shape[0]) - 2

        #plt.plot(x, chi_sqrds, c='b', alpha=0.2)
        plt.plot(x, chi_sqrds.mean(axis = 1), label=smart_label(args))

    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    plt.savefig('figs/result_{}.png'.format(filename.split('.')[0]))


if __name__ == '__main__':

    filenames = [
        'chi-sqrd_val.pt',
        'transition_val.pt',
        'similarity1_val.pt',
        'similarity2_val.pt',
        'mode_collapse_val.pt'
    ]

    for filename in filenames:
        plot_boxes(filename)

    predictor_name = 'C3_BREAST'
    event_name = 'I9_CHD'

    plot_survival(predictor_name, event_name)
    plot_hr(predictor_name, event_name)
