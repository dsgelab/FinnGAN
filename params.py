max_sequence_length = 2017 - 2000 + 1

endpoints = [
    'I9_HYPTENS', 
    'I9_ANGINA', 
    'I9_HEARTFAIL_NS', 
    'I9_STR_EXH', 
    'I9_CHD', 
    'C3_BREAST', 
    'I9_MI', 
    'L12_ATOPIC', 
    'K11_IBD_STRICT', 
    'G6_ALZHEIMER',
    'HYPOTHY_LEVOTHY',
    'J10_ASTHMA',
    'T2D'
]

use_default_params = True
use_optimal_bs_and_lr = False


# Generator params
mem_slots = 1
head_size = 6
embed_size = 4
temperature = 100
num_heads = 6
num_blocks = 4


# Discriminator params
n_embeddings = 3
embed_size = embed_size
out_channels = 10
filter_sizes = [2, 3, 4, 5] # values can be at most the sequence_length
n_critic = 1
use_aux_info = True 
use_mbd = True
mbd_out_features = 4
mbd_kernel_dims = 2

# Training params
batch_size = 128*16
#batch_size = 17
n_epochs = 30
print_step = 1#max(n_epochs // 20, 1)
lr = 5 * 10 ** (-5)
#lr = 10 ** (-6.864947029352897)




ignore_time = True
dummy_batch_size = 256
dummy_batch_size2 = 10000

ignore_similar = True
GAN_type = 'standard'

one_sided_label_smoothing = True
relativistic_average = False

params_name = 'general'
    
G_filename = 'models/{}_{}_{}_model.pt'.format(params_name, GAN_type, relativistic_average)


transition_score_mad = 0.539579
chi_squared_score_mad = 0.814368
indv_score_mad = 0.295352
similarity_score_mad = 0.212406


br_cancer_and_chd_params = {'target': -0.06987541913986206, 'params': {'batch_size': 27.443645721940197, 'embed_size': 8.382857459651621, 'head_size': 6.436161735123381, 'lr': 6.9150182741233275, 'mem_slots': 16.88672343316759, 'n_embeddings': 3.7334376616520126, 'num_blocks': 9.290433219377062, 'num_filters': 16.08094798893746, 'num_heads': 10.922316609867288, 'out_channels': 3.401020089212995, 'temperature': 533.0354183956473, 'n_critic': 1}}

general_params = [
    {'target': -0.752271831035614, 'params': {'batch_size': 87.22019450397207, 'embed_size': 8.047302668718405, 'head_size': 12.47825264723444, 'lr': 5.899279052642061, 'mem_slots': 8.62215146800973, 'n_embeddings': 7.5364329517898145, 'num_blocks': 6.042834883454295, 'num_filters': 15.221701112837422, 'num_heads': 10.56986320698104, 'out_channels': 19.669446192782384, 'temperature': 163.1922306120787, 'n_critic': 1}},
    {'target': -0.7686513662338257, 'params': {'GAN_type': 1, 'batch_size': 59, 'embed_size': 8, 'head_size': 12, 'lr': 5.120670633363621, 'mem_slots': 4, 'n_critic': 1, 'n_embeddings': 3, 'num_blocks': 5, 'num_filters': 4, 'num_heads': 6, 'out_channels': 12, 'relativistic_average': 1, 'temperature': 354.1343259187828}}
    
]

transition_params = [
    {'target': -0.018841540440917015, 'params': {'batch_size': 48.662565212045955, 'lr': 5.829709737290266, 'temperature': 664.256731903}},
     
]

general_params1 = [
    {'target': -0.5692319273948669, 'params': {'batch_size': 97.62833563120961, 'lr': 7.097719647690072, 'temperature': 112.2618950139476}},
    {'target': -7.569706916809082, 'params': {'batch_size': 85.89919785463678, 'lr': 4.4296372481419946, 'temperature': 90.93189011962716}}

]

GAN_types = ['standard', 'feature matching', 'wasserstein', 'least squares']
relativistic_average_options = [None, True, False]


plot_style = 'seaborn'
#plot_style = 'bmh'


if not use_default_params:
    if params_name == 'general':
        parameters = general_params[1]['params']
    elif params_name == 'br_cancer_and_chd':
        parameters = br_cancer_and_chd_params['params']

    batch_size = parameters['batch_size']
    embed_size = parameters['embed_size']
    head_size = parameters['head_size']
    lr = parameters['lr']
    mem_slots = parameters['mem_slots']
    n_embeddings = parameters['n_embeddings']
    num_blocks = parameters['num_blocks']
    num_filters = parameters['num_filters']
    num_heads = parameters['num_heads']
    out_channels = parameters['out_channels']
    temperature = parameters['temperature']
    n_critic = parameters['n_critic']

    batch_size = int(batch_size)
    embed_size = int(embed_size)
    head_size = int(head_size)
    mem_slots = int(mem_slots)
    n_embeddings = int(n_embeddings)
    num_blocks = int(num_blocks)
    num_filters = int(num_filters)
    num_heads = int(num_heads)
    out_channels = int(out_channels)
    n_critic = int(n_critic)

    filter_sizes = list(range(2, 2 + num_filters)) # values can be at most the sequence_length
    lr = 10 ** (-lr)
    
    if 'GAN_type' in parameters:
        GAN_type = GAN_types[int(parameters['GAN_type'])]
    if 'relativistic_average' in parameters:
        relativistic_average = relativistic_average_options[int(parameters['relativistic_average'])]
    
if use_optimal_bs_and_lr:
    if params_name == 'general':
        parameters = general_params1[-1]['params']
    elif params_name == 'transition':
        parameters = transition_params[-1]['params']

    batch_size = parameters['batch_size']
    lr = parameters['lr']
    temperature = parameters['temperature']
    
    batch_size = int(batch_size)
    