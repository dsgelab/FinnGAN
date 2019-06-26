import torch
import torch.nn as nn
import torch.nn.functional as F

class RelGANDiscriminator(nn.Module):
    def __init__(self, n_embeddings, vocab_size, embed_size, sequence_length, out_channels, filter_sizes):
        super(RelGANDiscriminator, self).__init__()
        self.n_embeddings = n_embeddings
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.sequence_length = sequence_length
        self.out_channels = out_channels
        self.filter_sizes = filter_sizes
        self.n_total_out_channels = self.out_channels * len(self.filter_sizes)
        
        self.embeddings = nn.ModuleList([nn.Embedding(self.vocab_size, self.embed_size) for _ in range(self.n_embeddings)])
        
        self.convolutions = nn.ModuleList([nn.Conv2d(1, self.out_channels, (filter_size, self.embed_size)) for filter_size in self.filter_sizes])
        
        self.hidden1 = nn.Linear(self.n_total_out_channels, self.n_total_out_channels // 2 + 1)
        self.hidden2 = nn.Linear(self.n_total_out_channels // 2 + 1, self.n_total_out_channels // 4 + 1)
        
        self.output_layer = nn.Linear(self.n_total_out_channels // 4 + 1, 1)
        
    def forward(self, x, return_mean = True):
        '''
            input:
                x (torch.FloatTensor): onehot of size [batch_size, self.sequence_length, self.vocab_size]
        '''
        hidden = []
        for embedding in self.embeddings:
            # using tensordot instead of matmul because matmul produces "UnsafeViewBackward" grad_fn
            embed = torch.tensordot(x, embedding.weight, dims = 1)
            embed = embed.unsqueeze(dim = 1) # Add channel dimension => shape: [batch_size, 1, self.sequence_length, self.embed_size]
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
        
        hidden = self.hidden1(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels // 2 + 1]
        hidden = F.relu(hidden)
        
        hidden = self.hidden2(hidden) # [batch_size, self.n_embeddings, self.n_total_out_channels // 4 + 1]
        hidden = F.relu(hidden)
        
        result = self.output_layer(hidden) # [batch_size, self.n_embeddings, 1]
        result = torch.sigmoid(result)
        
        result = result.squeeze(dim = -1) # [batch_size, self.n_embeddings]
        
        mean_result = torch.mean(result, dim = 1) # [batch_size, 1]
        
        if return_mean:
            return mean_result
        else:
            return result
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    