import torch
import torch.nn as nn
import torch.nn.functional as F
from minibatch_discrimination import MinibatchDiscrimination

cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class RelGANDiscriminator(nn.Module):
    def __init__(self, n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes, mbd_out_features, mbd_kernel_dims):
        super(RelGANDiscriminator, self).__init__()
        self.n_embeddings = n_embeddings
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.n_total_out_channels = self.out_channels * len(self.filter_sizes)
        self.hidden1_size = self.n_total_out_channels
        self.hidden2_size = self.n_total_out_channels // 2 + 1
        
        self.mbd_in_features = self.n_embeddings * self.hidden1_size
        self.mbd_out_features = mbd_out_features
        self.mbd_kernel_dims = mbd_kernel_dims
        
        self.hidden1_size += self.mbd_out_features + 2 + self.sequence_length * self.vocab_size
        
        self.embeddings = nn.ModuleList([nn.Embedding(self.vocab_size, self.embed_size) for _ in range(self.n_embeddings)])
        
        self.convolutions = nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(1, self.out_channels, (filter_size, self.embed_size + 1))) for filter_size in self.filter_sizes])
        
        self.MBD = MinibatchDiscrimination(self.mbd_in_features, self.mbd_out_features, self.mbd_kernel_dims)
        
        self.hidden1 = nn.utils.spectral_norm(nn.Linear(self.hidden1_size, self.hidden2_size))
        self.hidden2 = nn.utils.spectral_norm(nn.Linear(self.hidden2_size, self.n_total_out_channels // 4 + 1))
        
        self.output_layer = nn.utils.spectral_norm(nn.Linear(self.n_total_out_channels // 4 + 1, 1))
        
    def forward(self, x, age, sex, proportion, dist, return_mean = True, feature_matching = False, return_critic = False):
        '''
            input:
                x (torch.FloatTensor): onehot of size [batch_size, self.sequence_length, self.vocab_size]
        '''
        hidden = []
        
        ages = age.view(-1, 1).type(Tensor) + torch.arange(self.sequence_length).type(Tensor)#, dtype = torch.float32, device = device)
        ages /= 100
        
        sexes = (sex.view(-1, 1) - 2).repeat(1, self.n_embeddings).type(Tensor)
        
        proportion = torch.tensor(proportion).repeat(x.shape[0], self.n_embeddings).type(Tensor)
        
        dist = dist.view(1, 1, -1) # [1, 1, self.sequence_length * self.vocab_size]
        dist = dist.repeat(x.shape[0], self.n_embeddings, 1).type(Tensor) # [batch_size, self.n_embeddings, self.sequence_length * self.vocab_size]
        
        ages = ages.unsqueeze(dim = 1).unsqueeze(dim = -1) # [batch_size, 1, self.sequence_length, 1]
        sexes = sexes.unsqueeze(dim = -1) # [batch_size, self.n_embeddings, 1]
        proportion = proportion.unsqueeze(dim = -1) # [batch_size, self.n_embeddings, 1]
        
        for embedding in self.embeddings:
            # using tensordot instead of matmul because matmul produces "UnsafeViewBackward" grad_fn
            embed = torch.tensordot(x, embedding.weight, dims = 1)
            embed = embed.unsqueeze(dim = 1) # Add channel dimension => shape: [batch_size, 1, self.sequence_length, self.embed_size]
            embed = torch.cat([embed, ages], dim = -1) # [batch_size, 1, self.sequence_length, self.embed_size + 1]
            max_pools = []
            for i, convolution in enumerate(self.convolutions):
                conv = convolution(embed) # [batch_size, self.out_channels, self.sequence_length - self.filter_sizes[i] + 1, 1]
                out = F.relu(conv)
                max_pool = F.max_pool2d(out, (self.sequence_length - self.filter_sizes[i] + 1, 1)) # [batch_size, self.out_channels, 1, 1]
                max_pools.append(max_pool)
            max_pools = torch.cat(max_pools, dim = 1) # [batch_size, self.n_total_out_channels, 1, 1]
            hidden.append(max_pools)
            
        hidden = torch.cat(hidden, dim = -1) # [batch_size, self.n_total_out_channels, 1, self.n_embeddings]
        hidden = hidden.permute(0, 3, 1, 2).squeeze(dim = -1) # [batch_size, self.n_embeddings, self.n_total_out_channels]
        hidden = self.MBD(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels + self.mbd_out_features]
        
        hidden = torch.cat([hidden, sexes, proportion, dist], dim = -1) # [batch_size, self.n_embeddings, self.n_total_out_channels + self.mbd_out_features + 2 + self.sequence_length * self.vocab_size]
        features = hidden.view(hidden.shape[0], -1)
            
        if feature_matching:
            return features
        
        hidden = self.hidden1(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels // 2 + 1]
        hidden = F.relu(hidden)
        
        hidden = self.hidden2(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels // 4 + 1]
        hidden = F.relu(hidden)
        
        critic = self.output_layer(hidden) # [batch_size, self.n_embeddings, 1]
        critic = critic.squeeze(dim = -1) # [batch_size, self.n_embeddings]
        
        result = torch.sigmoid(critic)
        
        mean_critic = torch.mean(critic, dim = 1) # [batch_size, 1]
        mean_result = torch.mean(result, dim = 1) # [batch_size, 1]
        
        if return_mean:
            if not return_critic:
                output = mean_result
            else:
                output = mean_critic
        else:
            if not return_critic:
                output = result
            else:
                output = critic
            
        return output
    
    