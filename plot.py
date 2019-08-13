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
    
        

# TODO: make a loop of folders?
def main(use_aux_info, use_mbd, use_gp, n_endpoints):
    plt.style.use(plot_style)
    
    # TODO: add smart detection of folders without use_gp
    args = {
        'use_aux_info': use_aux_info,
        'use_mbd': use_mbd,
        #'use_gp': use_gp,
        'n_endpoints': n_endpoints,
    }
    
    dirname_parts = []
    
    for k, v in args.items():
        dirname_parts.append(k + '=' + str(v))
    
    dirnames = glob.glob('results/' + ' '.join(dirname_parts) + '/*/')
    
    chi_sqrds = []
    
    for seed_dir in dirnames:
        chi_sqrd = torch.load(seed_dir + 'chi-sqrd_train.pt').numpy()
        chi_sqrds.append(chi_sqrd)
        
    chi_sqrds = np.stack(chi_sqrds, axis = 1)
        
    x = np.arange(chi_sqrds.shape[0]) - 2

    plt.plot(x, chi_sqrds, c='b', alpha=0.2)
    plt.plot(x, chi_sqrds.mean(axis = 1), c='b', label=smart_label(args))
    plt.axvline(-1, color='k', linestyle='--', label='pretraining')
    plt.legend()
    
    plt.savefig('figs/result_chi_sqrd.jpeg')


if __name__ == '__main__':
    
    use_aux_info = True 
    use_mbd = True
    use_gp = True
    n_endpoints = 6
    #n_endpoints = len(endpoints)
    
    main(use_aux_info, use_mbd, use_gp, n_endpoints)