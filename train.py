import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, Iterator, Dataset, Example
from utils import *
from math import isnan


cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Define the generator pre-train function

def pretrain_generator(G, train, batch_size, vocab_size, sequence_length, n_epochs, lr, print_step = 10):
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    
    if cuda:
        G.cuda()
        loss_function.cuda()
    
    for e in range(n_epochs):
        train_iter = Iterator(train, batch_size = batch_size, device = device)
        loss_total = 0
        count = 0
        
        for batch in train_iter:
            train_data = batch.ENDPOINT.transpose(0, 1)
            train_data_one_hot = F.one_hot(train_data, vocab_size).type(Tensor)
            
            start_token = train_data[:, :1]
            optimizer.zero_grad()

            #memory = G.initial_state(batch_size = train_data.shape[0])

            if cuda:
                start_token = start_token.cuda()
                #memory = memory.cuda()
                
            logits, _, _ = G(start_token, batch.AGE, batch.SEX.view(-1), None, sequence_length, 1.0)

            loss = loss_function(logits, train_data_one_hot)
            
            loss_total += loss.item()
            count += 1

            loss.backward()
            optimizer.step()
            
        
        if e % print_step == 0:
            print(
                "[Epoch %d/%d] [G loss: %f]"
                % (e, n_epochs, loss_total / count)
            )

# Define the training function
            
# GAN_type is one of ['standard', 'feature matching', 'wasserstein', 'least squares']
# relativistic_average is one of [None, True, False]
def train_GAN(G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, print_step = 10, score_fn = get_scores, ignore_time = True, dummy_batch_size = 128, ignore_similar = True, one_sided_label_smoothing = True, relativistic_average = None):    
    scores = []
    accuracies_real = []
    accuracies_fake = []
    
    score = score_fn(G, ENDPOINT, train, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print('Scores before training:', *score)
    scores.append(score)
    
    print('pretraining generator...')
    pretrain_generator(G, train, batch_size, vocab_size, sequence_length, max(n_epochs // 10, 1), lr * 100, print_step = max(n_epochs // 10 - 1, 1))
    print('pretraining complete')
    
    score = score_fn(G, ENDPOINT, train, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print("[Scores:", *score, "]")
    scores.append(score)
    
    if GAN_type == 'standard':
        if relativistic_average is None:
            criterionG = criterionD = torch.nn.BCELoss()
        else:
            criterionG = criterionD = torch.nn.BCEWithLogitsLoss()
        
    elif GAN_type == 'least squares':
        criterionG = criterionD = torch.nn.MSELoss()
        
    elif GAN_type == 'feature matching':
        criterionG = torch.nn.MSELoss()
        if relativistic_average is None:
            criterionD = torch.nn.BCELoss()
        else:
            criterionD = torch.nn.BCEWithLogitsLoss()
        
    if GAN_type == 'wasserstein':
        optimizer_G = torch.optim.RMSprop(G.parameters(), lr=lr)
        optimizer_D = torch.optim.RMSprop(D.parameters(), lr=lr)
    else:
        optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
    
    if cuda:
        G.cuda()
        D.cuda()
        if GAN_type != 'wasserstein':
            criterionD.cuda()
            criterionG.cuda()
    
    for e in range(n_epochs):
        train_iter = Iterator(train, batch_size = batch_size, device = device)
        #loss_total = 0
        #count = 0
        
        for batch in train_iter:
            train_data = batch.ENDPOINT.transpose(0, 1)
            train_data_one_hot = F.one_hot(train_data, vocab_size).type(Tensor)

            start_token = train_data[:, :1]
            
            # Adversarial ground truths
            valid = Variable(Tensor(train_data.shape[0]).fill_(1.0), requires_grad=False) * (0.9 if one_sided_label_smoothing else 1) # one-sided label smoothing
            fake = Variable(Tensor(train_data.shape[0]).fill_(0.0), requires_grad=False)

            optimizer_G.zero_grad()

            # Generate a batch of images
            #memory = G.initial_state(batch_size = train_data.shape[0])
            #if cuda:
                #memory = memory.cuda()

            temp = temperature ** ((e + 1) / n_epochs)
            fake_one_hot, _, _ = G(start_token, batch.AGE, batch.SEX.view(-1), None, sequence_length, temp)

            # Loss measures generator's ability to fool the discriminator
            if GAN_type == 'feature matching':
                D_out_fake = D(fake_one_hot, batch.AGE, batch.SEX.view(-1), feature_matching = True).mean(dim = 0)
                D_out_real = D(train_data_one_hot.detach(), batch.AGE, batch.SEX.view(-1), feature_matching = True).mean(dim = 0)
                
            elif GAN_type == 'standard' and relativistic_average is None:
                D_out_fake = D(fake_one_hot, batch.AGE, batch.SEX.view(-1))
                D_out_real = D(train_data_one_hot.detach(), batch.AGE, batch.SEX.view(-1))

            elif GAN_type in ['least squares', 'wasserstein'] or relativistic_average is not None:
                D_out_fake = D(fake_one_hot, batch.AGE, batch.SEX.view(-1), return_critic = True)
                D_out_real = D(train_data_one_hot.detach(), batch.AGE, batch.SEX.view(-1), return_critic = True)
                
                
            if GAN_type == 'feature matching':
                g_loss = criterionG(D_out_fake, D_out_real)
                
            elif GAN_type in ['standard', 'least squares']:
                if relativistic_average is None:
                    g_loss = criterionG(D_out_fake, valid)
                elif relativistic_average:
                    g_loss = criterionG(D_out_fake - D_out_real.mean(0, keepdim=True), valid)
                else:
                    g_loss = criterionG(D_out_fake - D_out_real, valid)
                    
            elif GAN_type == 'wasserstein':
                if relativistic_average is None:
                    g_loss = -torch.mean(D_out_fake)
                elif relativistic_average:
                    g_loss = -torch.mean(D_out_fake - D_out_real.mean(0, keepdim=True))
                else:
                    g_loss = -torch.mean(D_out_fake - D_out_real)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            if GAN_type in ['least squares', 'wasserstein'] or relativistic_average is not None:
                D_out_real = D(train_data_one_hot, batch.AGE, batch.SEX.view(-1), return_critic = True).view(-1)
                D_out_fake = D(fake_one_hot.detach(), batch.AGE, batch.SEX.view(-1), return_critic = True).view(-1)
            else:
                D_out_real = D(train_data_one_hot, batch.AGE, batch.SEX.view(-1)).view(-1)
                D_out_fake = D(fake_one_hot.detach(), batch.AGE, batch.SEX.view(-1)).view(-1)
            
            
            if GAN_type in ['least squares', 'wasserstein'] or relativistic_average is not None:
                accuracy_real = torch.mean(torch.sigmoid(D_out_real))
                accuracy_fake = torch.mean(1 - torch.sigmoid(D_out_fake))
            else:
                accuracy_real = torch.mean(D_out_real)
                accuracy_fake = torch.mean(1 - D_out_fake)
            
            
            if GAN_type == 'wasserstein':
                if relativistic_average is None:
                    d_loss = -torch.mean(D_out_real) + torch.mean(D_out_fake)
                elif relativistic_average:
                    d_loss = -torch.mean(D_out_real - D_out_fake.mean(0, keepdim=True)) + torch.mean(D_out_fake - D_out_real.mean(0, keepdim=True))
                else:
                    d_loss = -torch.mean(D_out_real - D_out_fake) + torch.mean(D_out_fake - D_out_real)
                
            else:
                if relativistic_average is None:
                    real_loss = criterionD(D_out_real, valid)
                    fake_loss = criterionD(D_out_fake, fake)
                elif relativistic_average:
                    real_loss = criterionD(D_out_real - D_out_fake.mean(0, keepdim=True), valid)
                    fake_loss = criterionD(D_out_fake - D_out_real.mean(0, keepdim=True), fake)
                else:
                    real_loss = criterionD(D_out_real - D_out_fake, valid)
                    fake_loss = criterionD(D_out_fake - D_out_real, fake)
                d_loss = (real_loss + fake_loss) * 0.5

            d_loss.backward()
            optimizer_D.step()
            
            if GAN_type == 'wasserstein':
                # Clip weights of discriminator
                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01) # TODO: transform this into a tunable parameter

        if e % print_step == 0:
            print()
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f] [Acc real: %f] [Acc fake: %f]"
                % (e, n_epochs, d_loss.item(), g_loss.item(), accuracy_real, accuracy_fake)
            )
            score = score_fn(G, ENDPOINT, train, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
            print("[Scores:", *score, "]")
            scores.append(score)
            accuracies_real.append(accuracy_real)
            accuracies_fake.append(accuracy_fake)
            
    score = score_fn(G, ENDPOINT, train, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print('Scores after training:', *score)
    scores.append(score)
            
    output = [[] for _ in range(len(scores[0]))]
    for i in range(len(scores)):
        for j in range(len(scores[i])):
            output[j].append(scores[i][j])

    output.append(accuracies_real)
    output.append(accuracies_fake)
            
    for j in range(len(output)):
        output[j] = torch.stack(output[j])
            
    return tuple(output)
