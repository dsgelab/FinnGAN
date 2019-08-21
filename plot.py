import matplotlib.pyplot as plt
from utils import *
from params import *
import glob
import torch
from collections import OrderedDict

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

    color_list = ['#CF9821', '#98CF21', '#21CF98', '#CF2198', '#2198CF']

    pos = np.arange(1, len(data_dict) + 1)
    for p in pos:
        #parts = plt.violinplot(list(data_dict.values())[p - 1], [p], points=600, widths=0.7,
        #                showextrema=False, showmedians=False)
        #for pc in parts['bodies']:
        #    pc.set_alpha(1)
        plt.boxplot(
            list(data_dict.values())[p - 1],
            positions = [p],
            widths=0.4,
            patch_artist=True,
            boxprops = dict(facecolor=color_list[p - 1], alpha=0.7),
            medianprops = dict(color='k')
        )




    plt.xticks(pos, data_dict.keys())
    plt.title(filename.split('_val.pt')[0])

    plt.show()

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
    plt.savefig('figs/result_{}.svg'.format(filename.split('.')[0]))


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
