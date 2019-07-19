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

def train_GAN(G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, print_step = 10, score_fn = get_scores, ignore_time = True, dummy_batch_size = 128):    
    scores = []
    accuracies_real = []
    accuracies_fake = []
    
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, vocab_size, sequence_length)
    print('Scores before training:', *score)
    scores.append(score)
    
    print('pretraining generator...')
    pretrain_generator(G, train, batch_size, vocab_size, sequence_length, max(n_epochs // 10, 1), lr * 100, print_step = max(n_epochs // 10 - 1, 1))
    print('pretraining complete')
    
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, vocab_size, sequence_length)
    print("[Scores:", *score, "]")
    scores.append(score)
    
    criterionD = torch.nn.BCELoss()
    criterionG = torch.nn.MSELoss()
    
    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr)
    
    if cuda:
        G.cuda()
        D.cuda()
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
            valid = Variable(Tensor(train_data.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(train_data.shape[0]).fill_(0.0), requires_grad=False)

            optimizer_G.zero_grad()

            # Generate a batch of images
            #memory = G.initial_state(batch_size = train_data.shape[0])
            #if cuda:
                #memory = memory.cuda()

            temp = temperature ** ((e + 1) / n_epochs)
            fake_one_hot, _, _ = G(start_token, batch.AGE, batch.SEX.view(-1), None, sequence_length, temp)

            # Loss measures generator's ability to fool the discriminator
            _, D_features_fake = D(fake_one_hot, batch.AGE, batch.SEX.view(-1))
            _, D_features_real = D(train_data_one_hot.detach(), batch.AGE, batch.SEX.view(-1))
            g_loss = criterionG(D_features_fake.mean(dim = 0), D_features_real.mean(dim = 0))

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            D_out_real = D(train_data_one_hot, batch.AGE, batch.SEX.view(-1), feature_matching = False).view(-1)
            D_out_fake = D(fake_one_hot.detach(), batch.AGE, batch.SEX.view(-1), feature_matching = False).view(-1)
            
            #print(D_out_real)
            #print(torch.round(D_out_real))
            accuracy_real = torch.mean(D_out_real)
            accuracy_fake = torch.mean(1 - D_out_fake)
            
            real_loss = criterionD(D_out_real, valid * 0.9) # one-sided label smoothing
            fake_loss = criterionD(D_out_fake, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        if e % print_step == 0:
            print()
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f] [Acc real: %f] [Acc fake: %f]"
                % (e, n_epochs, d_loss.item(), g_loss.item(), accuracy_real, accuracy_fake)
            )
            score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, vocab_size, sequence_length)
            print("[Scores:", *score, "]")
            scores.append(score)
            accuracies_real.append(accuracy_real)
            accuracies_fake.append(accuracy_fake)
            
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, vocab_size, sequence_length)
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
