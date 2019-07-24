max_sequence_length = 5


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
dummy_batch_size = 128

ignore_similar = True
    
G_filename = 'models/model.pt'

transition_score_mad = 0.539579
chi_squared_score_mad = 0.814368
indv_score_mad = 0.295352
similarity_score_mad = 0.212406
