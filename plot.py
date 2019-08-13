import matplotlib.pyplot as plt
from utils import *
from params import *
import glob
import torch

def smart_label(args):
    false_count = 0
    true_count = 0
    
    for k, v in args.items():
        if k != 'n_endpoints':
            if v:
                true_count += 1
            else:
                false_count += 1
                
    if true_count == 1:
        for k, v in args.items():
            if k != 'n_endpoints' and v:
                return k + '=True'
    elif true_count == 0:
        return 'all=False'
    elif false_count == 0:
        return 'all=True'
    elif false_count == 1:
        for k, v in args.items():
            if k != 'n_endpoints' and not v:
                return k + '=False'
            
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
            res[k] = bool(v)
            
    if 'use_gp' not in res:
        res['use_gp'] = False
            
    return res
        

def main(filename, n_endpoints):
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

        plt.plot(x, chi_sqrds, c='b', alpha=0.2)
        plt.plot(x, chi_sqrds.mean(axis = 1), c='b', label=smart_label(args))
        plt.axvline(-1, color='k', linestyle='--', label='pretraining')
        plt.legend()

        plt.savefig('figs/result_chi_sqrd.jpeg')


if __name__ == '__main__':
    
    n_endpoints = 6
    #n_endpoints = len(endpoints)
    
    filename = 'chi-sqrd_train.pt'
    
    main(filename, n_endpoints)