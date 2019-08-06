import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, Iterator, Dataset, Example
from utils import *
from math import isnan
import numpy as np


cuda = torch.cuda.is_available()

# Try setting the device to a GPU
device = torch.device("cuda:0" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Define the generator pre-train function


            
def get_modified_batch(batch, ENDPOINT, n = 5):
    data = batch.ENDPOINT.transpose(0, 1).cpu()
    None_i = ENDPOINT.vocab.stoi['None']
    res = [data]
    
    for _ in range(n):
        tmp = torch.empty(data.shape).fill_(None_i).type(data.dtype)
        for indv_i, indv in enumerate(data):
            endpoints_not_None = indv[indv != None_i]
            
            if endpoints_not_None.shape[0] > 0:
                np_idx = np.sort(np.random.choice(np.arange(data.shape[1]), size = endpoints_not_None.shape[0], replace = False))
                idx = torch.from_numpy(np_idx)
                tmp[indv_i, idx] = endpoints_not_None
            
        res.append(tmp)
        
    res = torch.cat(res)
    
    ages = batch.AGE.repeat(n + 1)
    sexes = batch.SEX.view(-1).repeat(n + 1)
    
    if cuda:
        res = res.cuda()
        ages = ages.cuda()
        sexes = sexes.cuda()
    
    return res, ages, sexes
            
            

def pretrain_generator(G, train, batch_size, vocab_size, sequence_length, n_epochs, lr, ENDPOINT, print_step = 10):
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
            train_data, ages, sexes = get_modified_batch(batch, ENDPOINT)
            #train_data = batch.ENDPOINT.transpose(0, 1)
            train_data_one_hot = F.one_hot(train_data, vocab_size).type(Tensor)
            
            start_token = train_data[:, :1]
            optimizer.zero_grad()

            #memory = G.initial_state(batch_size = train_data.shape[0])

            if cuda:
                start_token = start_token.cuda()
                #memory = memory.cuda()
                
            logits, _, _ = G(start_token, ages, sexes, None, sequence_length, 1.0)

            loss = loss_function(logits, train_data_one_hot)
            
            loss_total += loss.item()
            count += 1

            loss.backward()
            optimizer.step()
            
        
        if (e + 1) % print_step == 0:
            print(
                "[Epoch %d/%d] [G loss: %f]"
                % (e, n_epochs, loss_total / count)
            )

            
            
# Define the training function
            
# GAN_type is one of ['standard', 'feature matching', 'wasserstein', 'least squares']
# relativistic_average is one of [None, True, False]
def train_GAN(G, D, train, val, ENDPOINT, batch_size, vocab_size, sequence_length, n_epochs, lr, temperature, GAN_type, n_critic, print_step = 10, score_fn = get_scores, ignore_time = True, dummy_batch_size = 128, ignore_similar = True, one_sided_label_smoothing = True, relativistic_average = None, searching = False):    
    scores_train = []
    scores_val = []
    accuracies_real = []
    accuracies_fake = []
    
    if not searching:
        score = score_fn(G, ENDPOINT, train, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
        print('Scores before training (train):', *score)
        scores_train.append(score)
    
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print('Scores before training (val):', *score)
    scores_val.append(score)
    
    print('pretraining generator...')
    pretrain_generator(G, train, batch_size, vocab_size, sequence_length, max(n_epochs // 10, 1), lr * 100, ENDPOINT, print_step = max(n_epochs // 10 - 1, 1))
    print('pretraining complete')
    
    if not searching:
        score = score_fn(G, ENDPOINT, train, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
        print("[Scores (train):", *score, "]")
        scores_train.append(score)
    
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print("[Scores (val):", *score, "]")
    scores_val.append(score)
    
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
    
    e = 0
    
    while e < n_epochs * n_critic:
        train_iter = Iterator(train, batch_size = batch_size, device = device)
        #loss_total = 0
        #count = 0
        
        for batch in train_iter:
            train_data, ages, sexes = get_modified_batch(batch, ENDPOINT)
            #train_data = batch.ENDPOINT.transpose(0, 1)
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

            temp = temperature ** ((e + 1) / (n_epochs * n_critic))
            fake_one_hot, fake_data, _ = G(start_token, ages, sexes, None, sequence_length, temp)
            
            proportion = train_data.unique(dim = 0).shape[0] / train_data.shape[0]
            proportion_fake = fake_data.unique(dim = 0).shape[0] / fake_data.shape[0]
            
            if e % n_critic == 0:
                # Loss measures generator's ability to fool the discriminator
                if GAN_type == 'feature matching':
                    D_out_fake = D(fake_one_hot, ages, sexes, proportion_fake, feature_matching = True).mean(dim = 0)
                    D_out_real = D(train_data_one_hot.detach(), ages, sexes, proportion, feature_matching = True).mean(dim = 0)

                elif GAN_type == 'standard' and relativistic_average is None:
                    D_out_fake = D(fake_one_hot, ages, sexes, proportion_fake)
                    D_out_real = D(train_data_one_hot.detach(), ages, sexes, proportion)

                elif GAN_type in ['least squares', 'wasserstein'] or relativistic_average is not None:
                    D_out_fake = D(fake_one_hot, ages, sexes, proportion_fake, return_critic = True)
                    D_out_real = D(train_data_one_hot.detach(), ages, sexes, proportion, return_critic = True)


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
                D_out_real = D(train_data_one_hot, ages, sexes, proportion, return_critic = True).view(-1)
                D_out_fake = D(fake_one_hot.detach(), ages, sexes, proportion_fake, return_critic = True).view(-1)
            else:
                D_out_real = D(train_data_one_hot, ages, sexes, proportion).view(-1)
                D_out_fake = D(fake_one_hot.detach(), ages, sexes, proportion_fake).view(-1)
            
            
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
                    p.data.clamp_(-0.01, 0.01) # TODO: transform this into a tunable parameter?

        if (e + 1) % (print_step * n_critic) == 0:
            print()
            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f] [Acc real: %f] [Acc fake: %f]"
                % (e, n_epochs * n_critic, d_loss.item(), g_loss.item(), accuracy_real, accuracy_fake)
            )
            if not searching:
                score = score_fn(G, ENDPOINT, train, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
                print("[Scores (train):", *score, "]")
                scores_train.append(score)
            
            score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
            print("[Scores (val):", *score, "]")
            scores_val.append(score)
            accuracies_real.append(accuracy_real)
            accuracies_fake.append(accuracy_fake)
            
            # TODO: detect convergence
            
        e += 1
            
    if not searching:
        score = score_fn(G, ENDPOINT, train, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
        print('Scores after training (train):', *score)
        scores_train.append(score)
    
    score = score_fn(G, ENDPOINT, val, dummy_batch_size, ignore_time, True, True, ignore_similar, vocab_size, sequence_length)
    print('Scores after training (val):', *score)
    scores_val.append(score)
            
    if not searching:
        output = [[] for _ in range(len(scores_train[0]) + len(scores_val[0]))]
        offset = len(scores_train[0])
    else:
        output = [[] for _ in range(len(scores_val[0]))]
        offset = 0
        
    if not searching:
        for i in range(len(scores_train)):
            for j in range(len(scores_train[i])):
                output[j].append(scores_train[i][j])
            
    for i in range(len(scores_val)):
        for j in range(len(scores_val[i])):
            output[j + offset].append(scores_val[i][j])

    output.append(accuracies_real)
    output.append(accuracies_fake)
            
    for j in range(len(output)):
        output[j] = torch.stack(output[j])
            
    return tuple(output)



if __name__ == '__main__':
    nrows = 1_000_000
    train, val, ENDPOINT, AGE, SEX, vocab_size, sequence_length, n_individuals = get_dataset(nrows = nrows)
    
    print(get_modified_batch(next(iter(Iterator(val, batch_size = 5))), ENDPOINT))
