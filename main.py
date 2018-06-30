import time
import math
from torch import optim
import torch.nn as nn

import random

from models.losses import JSDLoss
from utils.helpers import tensorsFromPair, tensorFromSentence, showPlot
from params import *

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainDiscriminator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length):
    for p in discriminator.parameters():
        p.requires_grad = True

    generator.eval()
    discriminator.train()
    generator.zero_grad()
    discriminator.zero_grad()

    free_run_nllloss, G_free_run_output, G_free_run_output_sents = generator(x=input_tensor,
                                                                             y=target_tensor,
                                                                             encoder_hidden=encoder_hidden,
                                                                             input_length=input_length,
                                                                             target_length=target_length,
                                                                             criterion=criterion,
                                                                             teacher_forced=False,
                                                                             train=True)

    teacher_forced_nllloss, G_teacher_forced_output, G_teacher_forced_output_sents = generator(x=input_tensor,
                                                                                                 y=target_tensor,
                                                                                                 encoder_hidden=encoder_hidden,
                                                                                                 input_length=input_length,
                                                                                                 target_length=target_length,
                                                                                                 criterion=criterion,
                                                                                                 teacher_forced=True,
                                                                                                 train=True)

    D_free_run_feat, D_free_run_out = discriminator(G_free_run_output.type(torch.LongTensor).to(device))
    D_teacher_forced_feat, D_teacher_forced_out = discriminator(G_teacher_forced_output.type(torch.LongTensor).to(device))

    D_loss = -(torch.mean(D_teacher_forced_out) - torch.mean(D_free_run_out))

    D_loss.backward(retain_graph=True)
    discriminator_optimizer.step()

    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

    return discriminator, discriminator_optimizer, D_loss.data.cpu().numpy()

def trainGenerator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length):
    for p in discriminator.parameters():
        p.requires_grad = False

    generator.train()
    discriminator.eval()
    generator.zero_grad()
    discriminator.zero_grad()

    free_run_nllloss, G_free_run_output, G_free_run_output_sents = generator(x=input_tensor,
                                                                             y=target_tensor,
                                                                             encoder_hidden=encoder_hidden,
                                                                             input_length=input_length,
                                                                             target_length=target_length,
                                                                             criterion=criterion,
                                                                             teacher_forced=False,
                                                                             train=True)

    teacher_forced_nllloss, G_teacher_forced_output, G_teacher_forced_output_sents = generator(x=input_tensor,
                                                                                                 y=target_tensor,
                                                                                                 encoder_hidden=encoder_hidden,
                                                                                                 input_length=input_length,
                                                                                                 target_length=target_length,
                                                                                                 criterion=criterion,
                                                                                                 teacher_forced=True,
                                                                                                 train=True)

    D_free_run_feat, D_free_run_out = discriminator(G_free_run_output.type(torch.LongTensor).to(device))
    D_teacher_forced_feat, D_teacher_forced_out = discriminator(G_teacher_forced_output.type(torch.LongTensor).to(device))
    #print(D_free_run_feat == D_teacher_forced_feat)
    #cos = nn.CosineSimilarity(dim=1)
    #print(cos(D_free_run_feat, D_teacher_forced_feat))
    #print(cos(D_free_run_feat, D_teacher_forced_feat).mean())
    

    free_run_loss = jsdloss(D_teacher_forced_feat, D_free_run_feat) / BATCH_SIZE
    teacher_forced_loss = jsdloss(D_free_run_feat, D_teacher_forced_feat) / BATCH_SIZE
    
    print('free_run_loss: ', free_run_loss)
    print('teacher_forced_loss: ', teacher_forced_loss)

    G_loss = (teacher_forced_nllloss / target_length) + free_run_loss# + teacher_forced_loss

    G_loss.backward(retain_graph=True)
    generator_optimizer.step()

    return generator, generator_optimizer, G_loss.data.cpu().numpy()[0][0], (free_run_nllloss.item() / target_length)

def train(input_tensor, target_tensor, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, max_length=MAX_LENGTH):
    encoder_hidden = generator.initEncoderHidden(batch_size=BATCH_SIZE)

    generator.zero_grad()

    input_tensor = torch.cat([input_tensor_i.unsqueeze(0) for input_tensor_i in input_tensor], 0).to(device)
    target_tensor = torch.cat([target_tensor_i.unsqueeze(0) for target_tensor_i in target_tensor], 0).to(device)

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    input_tensor = input_tensor.permute(1,0,2)
    target_tensor = target_tensor.permute(1,0,2)

    D_loss = 0
    for _ in range(D_PRESTEP):
        discriminator, discriminator_optimizer, D_loss_step = trainDiscriminator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length)
        D_loss += D_loss_step
    D_loss = D_loss/D_PRESTEP

    generator, generator_optimizer, G_loss, performance_nllloss = trainGenerator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length)

    return generator, generator_optimizer, discriminator, discriminator_optimizer, D_loss, G_loss, performance_nllloss

def trainIters(generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, training_pairs, n_iters, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    plot_G_losses = []
    plot_D_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_G_loss_total = 0
    plot_G_loss_total = 0
    print_D_loss_total = 0
    plot_D_loss_total = 0
    
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[(iter-1)*BATCH_SIZE:iter*BATCH_SIZE] # n tensor (list)
        input_tensor = [pair[0] for pair in training_pair]
        target_tensor = [pair[1] for pair in training_pair]

        generator, generator_optimizer, discriminator, discriminator_optimizer, D_loss, G_loss, loss = train(input_tensor, target_tensor, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss)
        print_loss_total += loss
        plot_loss_total += loss
        
        print_G_loss_total += G_loss
        plot_G_loss_total += G_loss
        
        print_D_loss_total += D_loss
        plot_D_loss_total += D_loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_G_loss_avg = print_G_loss_total / print_every
            print_G_loss_total = 0
            print_D_loss_avg = print_D_loss_total / print_every
            print_D_loss_total = 0
            print('%s (%d %d%%) NLLLoss: %.4f - Generator Loss: %.4f - Discriminator Loss: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, print_G_loss_avg, print_D_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
            plot_G_loss_avg = plot_G_loss_total / plot_every
            plot_G_losses.append(plot_G_loss_avg)
            plot_G_loss_total = 0
            
            plot_D_loss_avg = plot_D_loss_total / plot_every
            plot_D_losses.append(plot_D_loss_avg)
            plot_D_loss_total = 0

    return generator, discriminator, plot_losses, plot_G_losses, plot_D_losses


def evaluate(generator, sentence, input_lang, max_length=MAX_LENGTH):
    generator.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)
        input_length = input_tensor.size()[0]
        encoder_hidden = generator.initEncoderHidden(batch_size=input_tensor.shape[1])

        G_free_run_output_sents, _ = generator(x=input_tensor,
                                                     y=None,
                                                     encoder_hidden=encoder_hidden,
                                                     input_length=input_length,
                                                     target_length=0,
                                                     criterion=None,
                                                     teacher_forced=False,
                                                     train=False)

        return G_free_run_output_sents#, decoder_attentions

def evaluateRandomly(generator, input_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(generator, pair[0], input_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
