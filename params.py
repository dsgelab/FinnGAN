max_sequence_length = 5

endpoints = ['I9_HYPTENS', 'I9_ANGINA', 'I9_HEARTFAIL_NS', 'I9_STR_EXH', 'I9_CHD', 'C3_BREAST']


# Generator params
mem_slots = 1
head_size = 6
embed_size = 10
temperature = 5
num_heads = 10
num_blocks = 6


# Discriminator params
n_embeddings = 5
embed_size = embed_size
out_channels = 15
filter_sizes = [2, 3, 4] # values can be at most the sequence_length
n_critic = 1

# Training params
batch_size = 128
n_epochs = 10
print_step = max(n_epochs // 10, 1)
lr = 1e-4

'''
batch_size = 96
#chi-squared_score = 0.029918
embed_size = 8
head_size = 20
#indv_score = 0.124354
lr = 1e-5
mem_slots = 11
n_embeddings = 6
num_blocks = 3
num_filters = 8
num_heads = 17
out_channels = 12
#similarity_score = 0.077522
temperature = 11
#transition_score = 0.431095
'''

ignore_time = True
dummy_batch_size = 256

ignore_similar = True
GAN_type = 'feature matching'

one_sided_label_smoothing = True
relativistic_average = None

params_name = 'general'
    
G_filename = 'models/{}_{}_{}_model.pt'.format(params_name, GAN_type, relativistic_average)


transition_score_mad = 0.539579
chi_squared_score_mad = 0.814368
indv_score_mad = 0.295352
similarity_score_mad = 0.212406


br_cancer_and_chd_params = {'target': -0.06987541913986206, 'params': {'batch_size': 27.443645721940197, 'embed_size': 8.382857459651621, 'head_size': 6.436161735123381, 'lr': 6.9150182741233275, 'mem_slots': 16.88672343316759, 'n_embeddings': 3.7334376616520126, 'num_blocks': 9.290433219377062, 'num_filters': 16.08094798893746, 'num_heads': 10.922316609867288, 'out_channels': 3.401020089212995, 'temperature': 533.0354183956473, 'n_critic': 1}}

general_params = {'target': -0.752271831035614, 'params': {'batch_size': 87.22019450397207, 'embed_size': 8.047302668718405, 'head_size': 12.47825264723444, 'lr': 5.899279052642061, 'mem_slots': 8.62215146800973, 'n_embeddings': 7.5364329517898145, 'num_blocks': 6.042834883454295, 'num_filters': 15.221701112837422, 'num_heads': 10.56986320698104, 'out_channels': 19.669446192782384, 'temperature': 163.1922306120787, 'n_critic': 1}}


plot_style = 'seaborn'
#plot_style = 'bmh'

