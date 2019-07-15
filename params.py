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
batch_size = 64
n_epochs = 10
print_step = max(n_epochs // 10, 1)
lr = 1e-4


dummy_batch_size = 128

