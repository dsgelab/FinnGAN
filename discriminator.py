import torch
import torch.nn as nn
import torch.nn.functional as F

class RelGANDiscriminator(nn.Module):
    def __init__(self, n_embeddings, vocab_size, embed_size):
        super(RelGANDiscriminator, self).__init__()
        self.n_embeddings = n_embeddings
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embeddings = nn.ModuleList([nn.Embedding(self.vocab_size, self.embed_size) for _ in range(self.n_embeddings)])
        
    def forward(self, x):
        # TODO: The following might not propagate gradients
        indices = torch.matmul(x.type(torch.LongTensor), torch.tensor( list(range(self.vocab_size)) ).view(-1, 1))
        return indices