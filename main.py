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

    D_free_run_feat, D_free_run_out = discriminator(G_free_run_output.type(torch.LongTensor))
    D_teacher_forced_feat, D_teacher_forced_out = discriminator(G_teacher_forced_output.type(torch.LongTensor))

    D_loss = -(torch.mean(D_teacher_forced_out) - torch.mean(D_free_run_out))

    D_loss.backward(retain_graph=True)
    discriminator_optimizer.step()

    for p in discriminator.parameters():
        p.data.clamp_(-0.01, 0.01)

    return discriminator, D_loss

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

    D_free_run_feat, D_free_run_out = discriminator(G_free_run_output.type(torch.LongTensor))
    D_teacher_forced_feat, D_teacher_forced_out = discriminator(G_teacher_forced_output.type(torch.LongTensor))

    free_run_loss = jsdloss(D_teacher_forced_feat, D_free_run_feat)
    teacher_forced_loss = jsdloss(D_free_run_feat, D_teacher_forced_feat)

    G_loss = (teacher_forced_nllloss / target_length) + free_run_loss + teacher_forced_loss

    G_loss.backward(retain_graph=True)
    generator_optimizer.step()

    return generator, G_loss, (free_run_nllloss.item() / target_length)

def train(input_tensor, target_tensor, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, max_length=MAX_LENGTH):
    encoder_hidden = generator.initEncoderHidden(batch_size=BATCH_SIZE)

    generator.zero_grad()

    input_tensor = torch.cat([input_tensor_i.unsqueeze(0) for input_tensor_i in input_tensor], 0)
    target_tensor = torch.cat([target_tensor_i.unsqueeze(0) for target_tensor_i in target_tensor], 0)

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)

    input_tensor = input_tensor.permute(1,0,2)
    target_tensor = target_tensor.permute(1,0,2)

    for _ in range(5):
        discriminator, D_loss = trainDiscriminator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length)

    generator, G_loss, performance_nllloss = trainGenerator(input_tensor, target_tensor, encoder_hidden, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, input_length, target_length)

    return performance_nllloss

def trainIters(generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss, training_pairs, n_iters, print_every=1000, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[(iter-1)*BATCH_SIZE:iter*BATCH_SIZE] # n tensor (list)
        input_tensor = [pair[0] for pair in training_pair]
        target_tensor = [pair[1] for pair in training_pair]

        loss = train(input_tensor, target_tensor, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, jsdloss)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    return plot_losses


def evaluate(generator, sentence, input_lang, max_length=MAX_LENGTH):
    generator.eval()
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
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
