import torch
import torch.nn as nn
import torch.nn.functional as F
from minibatch_discrimination import MinibatchDiscrimination

cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class RelGANDiscriminator(nn.Module):
    def __init__(self, n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes, use_aux_info, use_mbd, mbd_out_features, mbd_kernel_dims):
        super(RelGANDiscriminator, self).__init__()
        self.n_embeddings = n_embeddings
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.n_total_out_channels = self.out_channels * len(self.filter_sizes)
        self.hidden1_size = self.n_total_out_channels + 1
        self.use_aux_info = use_aux_info
        self.use_mbd = use_mbd
        
        self.mbd_in_features = self.n_embeddings * self.hidden1_size
        self.mbd_out_features = mbd_out_features
        self.mbd_kernel_dims = mbd_kernel_dims
        
        if self.use_mbd:
            self.hidden1_size += self.mbd_out_features
        
        if self.use_aux_info:
            self.hidden1_size += 1 + self.sequence_length * self.vocab_size# + (self.vocab_size - 3) ** 2
        
        self.hidden2_size = self.hidden1_size // 2 + 1

        
        self.embeddings = nn.ModuleList([nn.Embedding(self.vocab_size, self.embed_size) for _ in range(self.n_embeddings)])
        
        self.convolutions = nn.ModuleList([nn.utils.spectral_norm(nn.Conv2d(1, self.out_channels, (filter_size, self.embed_size + 1))) for filter_size in self.filter_sizes])
        
        if self.use_mbd:
            self.MBD = MinibatchDiscrimination(self.mbd_in_features, self.mbd_out_features, self.mbd_kernel_dims)
        
        self.hidden1 = nn.utils.spectral_norm(nn.Linear(self.hidden1_size, self.hidden2_size))
        self.hidden2 = nn.utils.spectral_norm(nn.Linear(self.hidden2_size, self.hidden2_size // 2 + 1))
        
        self.output_layer = nn.utils.spectral_norm(nn.Linear(self.hidden2_size // 2 + 1, 1))
        
        
        
    def forward(self, x, age, sex, proportion, dist, embeds = None, return_mean = True, feature_matching = False, return_critic = False):
        '''
            input:
                x (torch.FloatTensor): onehot of size [batch_size, self.sequence_length, self.vocab_size]
        '''
        hidden = []
        
        ages = age.view(-1, 1).type(Tensor) + torch.arange(self.sequence_length).type(Tensor)#, dtype = torch.float32, device = device)
        ages /= 100
        
        sexes = (sex.view(-1, 1) - 2).repeat(1, self.n_embeddings).type(Tensor)
        
        ages = ages.unsqueeze(dim = 1).unsqueeze(dim = -1) # [batch_size, 1, self.sequence_length, 1]
        sexes = sexes.unsqueeze(dim = -1) # [batch_size, self.n_embeddings, 1]
        
        if self.use_aux_info:
            if isinstance(proportion, float):
                proportion = torch.tensor(proportion)
            proportion = proportion.view(-1, 1).expand(x.shape[0], self.n_embeddings).type(Tensor)
            proportion = proportion.unsqueeze(dim = -1) # [batch_size, self.n_embeddings, 1]

            dist = dist.view(-1, 1, self.sequence_length * self.vocab_size) # [1, 1, self.sequence_length * self.vocab_size]
            dist = dist.expand(x.shape[0], self.n_embeddings, -1).type(Tensor) # [batch_size, self.n_embeddings, self.sequence_length * self.vocab_size]
            
            #transition = transition.view(1, 1, -1) # [1, 1, (self.vocab_size - 3) ** 2]
            #transition = transition.repeat(x.shape[0], self.n_embeddings, 1).type(Tensor) # [batch_size, self.n_embeddings, (self.vocab_size - 3) ** 2]
            
        if embeds is None:
            embeddings = self.embeddings
        else:
            embeddings = embeds # [self.n_embeddings, batch_size, self.sequence_length, self.embed_size]
        
        for embed in embeddings:
            if embeds is None:
                # using tensordot instead of matmul because matmul produces "UnsafeViewBackward" grad_fn
                embed = torch.tensordot(x, embed.weight, dims = 1)
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
        
        hidden = torch.cat([hidden, sexes], dim = -1) # [batch_size, self.n_embeddings, self.n_total_out_channels + 1]
        
        if self.use_mbd:
            hidden = self.MBD(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels + 1 + self.mbd_out_features]
        
        if self.use_aux_info:
            hidden = torch.cat([hidden, proportion, dist], dim = -1) # [batch_size, self.n_embeddings, self.n_total_out_channels + 1 (+ self.mbd_out_features) + 1 + self.sequence_length * self.vocab_size]# + (self.vocab_size - 3) ** 2]
        
        features = hidden.view(hidden.shape[0], -1)
            
        if feature_matching:
            return features
        
        hidden = self.hidden1(hidden) # [batch_size, self.n_embeddings, self.hidden2_size]
        hidden = F.relu(hidden)
        
        hidden = self.hidden2(hidden) # [batch_size, self.n_embeddings, self.hidden2_size // 2 + 1]
        hidden = F.relu(hidden)
        
        critic = self.output_layer(hidden) # [batch_size, self.n_embeddings, 1]
        critic = critic.squeeze(dim = -1) # [batch_size, self.n_embeddings]
        
        result = torch.sigmoid(critic)
        
        if return_mean:
            mean_critic = torch.mean(critic, dim = 1) # [batch_size, 1]
            mean_result = torch.mean(result, dim = 1) # [batch_size, 1]
        
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
    
    