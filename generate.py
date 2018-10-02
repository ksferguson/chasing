###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--warmup', action='store_true',
                    help='use warmup text')
parser.add_argument('--warmupf', type=str, default='warmup.txt',
                    help='text file for warmup text')
parser.add_argument('--context', type=str, default='',
                    help='input string of words to set prior context')
parser.add_argument('--window', type=int, default=3785,
                    help='pointer window length')
parser.add_argument('--theta', type=float, default=0.6625523432485668,
                    help='mix between uniform distribution and pointer softmax distribution over previous words')
parser.add_argument('--lambdasm', type=float, default=0.12785920428335693,
                    help='linear mix between only pointer (1) and only vocab (0) distribution')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

def one_hot(idx, size, cuda=True):
    a = np.zeros((1, size), np.float32)
    a[0][idx] = 1
    v = Variable(torch.from_numpy(a))
    if cuda: v = v.cuda()
    return v



with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

print(args.cuda)
print(args.context)

#warm up routine
if args.warmup:
    print('** warmup **')
    warmup_str = ''
    with open(args.warmupf) as warmupf:
        warmup_data = warmupf.read()
        warmup_words = warmup_data.split() + ['<eos>']
        for word in warmup_words:
            warmup_str = warmup_str + ' ' + word
            warmup_word_idx = corpus.dictionary.word2idx[word]
            input.data.fill_(warmup_word_idx)
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word = corpus.dictionary.idx2word[word_idx]

#context routine
context_str = ''
pred_str = ''

if len (args.context) > 0:
    print('** start context **')

    next_word_history = None
    pointer_history = None
    context_words = args.context.split()
    for word in context_words:
        context_str = context_str + ' ' + word
        #fix: lookup word if not in corpus
        context_word_idx = corpus.dictionary.word2idx[word]
        input.data.fill_(context_word_idx)
        output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)
        if args.window > 0:
            #while stepping thorugh context, record words + hidden states
            rnn_out = rnn_outs[-1].squeeze(0)
            output_flat = output.view(-1, ntokens)
            next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
            #print(next_word_history.size())
            pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])
            #print(pointer_history.size())
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        word2 = corpus.dictionary.idx2word[word_idx]
        pred_str = pred_str + ' ' + word2
    print (context_str)
    print (pred_str)


pred_str = ''
print('** start pred w ptr **')
with open(args.outf, 'w') as outf:
    for i in range(args.words):
        output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)
        if args.window > 0:
            #while stepping thorugh preds, record words + hidden states
            rnn_out = rnn_outs[-1].squeeze(0)
            output_flat = output.view(-1, ntokens)
            #print(output_flat.size())
            next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
            #print(next_word_history.size())
            pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])
            #print(pointer_history.size())
            # pointer_history - fixup code below to calc pointer probs:
            # from pointer.py
            softmax_output_flat = torch.nn.functional.softmax(output_flat)
            theta = args.theta
            lambdah = args.lambdasm
            for idx, vocab_loss in enumerate(softmax_output_flat):
                p = vocab_loss
            #     if start_idx + idx > window:
                valid_next_word = next_word_history
                valid_pointer_history = pointer_history
                logits = torch.mv(valid_pointer_history, rnn_out.data.view(-1,1).squeeze())
                ptr_attn = torch.nn.functional.softmax(theta * Variable(logits)).view(-1, 1)
                ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
                p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss
            #     ###
            #     target_loss = p[targets[idx].data]
            #     loss += (-torch.log(target_loss)).data[0]

        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(p.data, 1)[0]
        #word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]

        pred_str = pred_str + ' ' + word

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))

print (pred_str)
