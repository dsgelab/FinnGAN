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
        'only_pretraining': 'MLE',
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

    if 'only_pretraining' not in res:
        res['only_pretraining'] = False

    return res

def data_dict_key(item):
    return np.median(item)

def plot_boxes(filename, n_endpoints):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    data_dict = {}

    for dirname in dirnames:
        args = get_args(dirname)

        if args['n_endpoints'] == n_endpoints:
            subdirnames = glob.glob(dirname + '/*/')

            tmp_data = []

            for seed_dir in subdirnames:
                tmp = torch.load(seed_dir + filename).numpy()
                tmp_data.append(tmp)

            tmp_data = np.stack(tmp_data, axis = 1)

            if get_true_count(args) <= 1:
                data_dict[smart_label(args)] = tmp_data[-1, :]

    #data_dict['MLE'] = tmp_data[1, :] # MLE/pretraining

    #data_dict = sorted(data_dict.items(), key=lambda kv: np.median(kv[1]))
    #data_dict = OrderedDict(data_dict)

    if n_endpoints == 6:
        color_list = ['#CF9821', '#98CF21', '#21CF98', '#CF2198', '#2198CF', '#9821CF']
        keys = ['BASE', 'MLE', 'MBD', 'FM', '0-GP', 'AUX']
    elif n_endpoints == 13:
        color_list = ['#98CF21', '#CF2198']
        keys = ['MLE', 'FM']

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
    plt.savefig('figs/{}_'.format(n_endpoints) + title + '.png')
    plt.clf()


def plot_hr(predictor_name, event_name, n_endpoints):
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

        if args['n_endpoints'] == n_endpoints:
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
                plt.savefig('figs/{}_'.format(n_endpoints) + title + '.png')
            plt.clf()



def plot_survival(predictor_name, event_name, n_endpoints):
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

        if args['n_endpoints'] == n_endpoints:
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
                plt.savefig('figs/{}_{}_survival_{}->{}.png'.format(n_endpoints, smart_label(args), predictor_name, event_name))
            plt.clf()


def plot_chi_sqrd_boxes_without_None(n_endpoints):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/')

    data_dict = {}

    for dirname in dirnames:
        args = get_args(dirname)

        if args['n_endpoints'] == n_endpoints:
            subdirnames = glob.glob(dirname + '/*/')

            tmp_data = []

            for seed_dir in subdirnames:
                freqs = torch.load(seed_dir + 'freqs.pt')
                freqs_fake = torch.load(seed_dir + 'freqs_fake.pt')
                tmp = chi_sqrd_dist(freqs[1:], freqs_fake[1:]).numpy()
                tmp_data.append(tmp)

            tmp_data = np.stack(tmp_data)

            if get_true_count(args) <= 1: #and (args['use_gp'] or args['feature_matching']):
                data_dict[smart_label(args)] = tmp_data

    if n_endpoints == 6:
        color_list = ['#CF9821', '#98CF21', '#21CF98', '#CF2198', '#2198CF', '#9821CF']
        keys = ['BASE', 'MLE', 'MBD', 'FM', '0-GP', 'AUX']
    elif n_endpoints == 13:
        color_list = ['#98CF21', '#CF2198']
        keys = ['MLE', 'FM']
    # color_list = ['#CF2198', '#2198CF']
    # keys = ['FM', '0-GP']

    pos = np.arange(1, len(data_dict) + 1)
    for p in pos:
        plt.boxplot(
            data_dict[keys[p - 1]],
            positions = [p],
            widths=0.4,
            patch_artist=True,
            boxprops = dict(facecolor=color_list[p - 1], alpha=1),
            medianprops = dict(color='k')
        )


    plt.xticks(pos, keys)
    title = 'Chi-squared distances without "None"'
    #plt.title(title)
    #plt.show()
    plt.ylabel(title)
    plt.savefig('figs/{}_'.format(n_endpoints) + title + '.png')
    plt.clf()




def main(n_endpoints):
    plt.style.use('classic')
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/0/')

    endpoints = [
        'None',
        'I9_HEARTFAIL_NS',
        'I9_HYPTENS',
        'I9_CHD',
        'I9_ANGINA',
        'I9_STR_EXH',
        'C3_BREAST',
    ]

    for dirname in dirnames:
        args = get_args(dirname)

        if args['n_endpoints'] == n_endpoints:
            freqs = torch.load(dirname + 'freqs.pt')
            freqs_fake = torch.load(dirname + 'freqs_fake.pt')

            chi_sqrd_d = chi_sqrd_dist(freqs, freqs_fake)

            freqs = freqs.numpy()
            freqs_fake = freqs_fake.numpy()

            x = np.arange(len(endpoints))

            fig, ax = plt.subplots(figsize=(6,6))

            width = 0.35
            ax.bar(x, freqs, width, bottom=0, label='real')
            ax.bar(x + width, freqs_fake, width, bottom=0, label='generated')
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(endpoints, rotation=90)
            ax.legend()

            ax.autoscale_view()
            fig.subplots_adjust(bottom=0.36)

            plt.savefig('figs/{}_freqs_{}_{}.png'.format(n_endpoints, smart_label(args), chi_sqrd_d.item()))
            plt.clf()

def plot_training(n_endpoints):
    plt.style.use(plot_style)

    dirnames = glob.glob('results/*/0/')

    for dirname in dirnames:
        args = get_args(dirname)

        if args['n_endpoints'] == n_endpoints:
            data = torch.load(dirname + 'chi-sqrd_val.pt').numpy()

            plt.plot(data)
            plt.xlabel('Training iterations')
            plt.ylabel('Metric value')

            plt.savefig('figs/{}_training_{}.png'.format(n_endpoints, smart_label(args)))
            plt.clf()

def plot_transition_matrices():
    n_endpoints = 6
    dirnames = glob.glob('results/*/0/')

    endpoints = [
        'I9_HEARTFAIL_NS',
        'I9_HYPTENS',
        'I9_CHD',
        'I9_ANGINA',
        'I9_STR_EXH',
        'C3_BREAST',
    ]

    plt.style.use('classic')

    for dirname in dirnames:
        args = get_args(dirname)

        if args['n_endpoints'] == n_endpoints:
            transition_freq_real = torch.load(dirname + 'transition_matrix_real_val.pt')
            transition_freq_fake = torch.load(dirname + 'transition_matrix_fake_val.pt')
            transition = torch.load(dirname + 'transition_val.pt')

            fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
            fig.subplots_adjust(left=0.22075, right=0.9)
            ticks = np.arange(n_endpoints)
            labels = endpoints
            cmap = 'plasma'

            vmax = torch.max(transition_freq_fake.max(), transition_freq_real.max())

            im = ax[0].matshow(transition_freq_real, vmin=0, vmax=vmax, cmap=cmap)
            ax[0].set_xticks(ticks)
            ax[0].set_xticklabels(labels, rotation=90)
            ax[0].set_title('Real', y = -0.2)

            im = ax[1].matshow(transition_freq_fake, vmin=0, vmax=vmax, cmap=cmap)
            ax[1].set_xticks(ticks)
            ax[1].set_xticklabels(labels, rotation=90)
            ax[1].set_title('Generated', y = -0.2)

            im = ax[2].matshow(torch.abs(transition_freq_fake - transition_freq_real), vmin=0, vmax=vmax, cmap=cmap)
            ax[2].set_xticks(ticks)
            ax[2].set_xticklabels(labels, rotation=90)
            ax[2].set_title('Abs. difference', y = -0.2)

            plt.yticks(ticks, labels)

            fig.colorbar(im, ax=ax.ravel().tolist(), ticks=np.linspace(0, vmax, 5), shrink = 0.27, aspect = 10)
            fig.suptitle('Transition probabilities (Transition metric: {})'.format(round_to_n(transition[-1].item(), 3)))
            fig.savefig('figs/{}_transition_matrices.png'.format(smart_label(args)))

            plt.clf()



if __name__ == '__main__':
    n_endpoints = 6

    plot_training(n_endpoints)

    plot_chi_sqrd_boxes_without_None(n_endpoints)

    filenames = [
        'chi-sqrd_val.pt',
        'transition_val.pt',
        'similarity1_val.pt',
        'similarity2_val.pt',
        'mode_collapse_val.pt'
    ]

    for filename in filenames:
        plot_boxes(filename, n_endpoints)

    predictor_name = 'I9_CHD'
    event_name = 'I9_HEARTFAIL_NS'

    plot_survival(predictor_name, event_name, n_endpoints)
    plot_hr(predictor_name, event_name, n_endpoints)

    plot_transition_matrices()

    main(n_endpoints)
