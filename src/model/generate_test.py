###############################################################################
# Language Modeling for Rare Words
#
# This file initially cloned from:
# https://github.com/salesforce/awd-lstm-lm.git
# Tag: PyTorch==0.1.12
# See LICENSE_awd-lstm-lm for original LICENSE
#
###############################################################################
import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data

import streamlit as st

from nltk.translate.bleu_score import corpus_bleu




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
parser.add_argument('--params-test', action='store_true',
                    help='execute a pointer cache params test')
parser.add_argument('--contextf', type=str, default='context.txt',
                    help='text file for context text')
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

#warm up routine
if args.warmup:
    print('** warming up model **')
    warmup_str = ''
    with open(args.warmupf) as warmupf:
        warmup_data = warmupf.read()
        warmup_words = warmup_data.split()
        for word in warmup_words:
            preprocessed_word = word
            warmup_str = warmup_str + ' ' + preprocessed_word
            warmup_word_idx = corpus.dictionary.word2idx[word]
            input.data.fill_(warmup_word_idx)
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word = corpus.dictionary.idx2word[word_idx]


st.write('# Parameter Testing')
st.write('This report documents the parameter testing for the pointer cache'
+ '. ' +
'The pointer cache takes two parameters: theta and lamba. This routine ' +
'(generate_test.py) iterates over the range of possible parameters and runs a' +
'set of contexts with 1-4 words deleted from the context. The BLEU score is '+
'then calculated to find the best parameter set.')

if args.params_test:
    with open(args.contextf) as contextf:
        contextf_data = contextf.readlines()
        st.write ('Reading context file: ')
        contextf_lines = [lines for lines in enumerate(contextf_data)]
        test_lines = []
        for context_line in contextf_lines:
            st.write (context_line[1].split())
            num_words_predict = np.random.random_integers(1,4)
            test_lines.append([context_line[1].split()
                , num_words_predict
                , context_line[1].split()[0:len(context_line[1].split())-num_words_predict]])
        for test_line in test_lines:
            st.write (test_line)

            st.write ('')
            st.write ('begin testing')
        for lambdasm_val_raw in range (6, 8):
            lambdasm_val = lambdasm_val_raw/10
            for theta_val_raw in range (1, 3):
                theta_val = theta_val_raw/10

                st.write('lambda: ', lambdasm_val, ', theta: ', theta_val)

                BLEU_score = 0.0

                # 3. iterate over params options lambdasm & theta
                # 4. for each params combo
                # 4.1 output params
                # 4.2 cycle through contexts, predicting missing words
                # 4.3 compare context to generated, calc BLEU score

                for test_line in test_lines:

                    full_context = test_line[0]
                    test_context = test_line[2]
                    test_preds = []
                    num_words_context = len(test_line[2])
                    num_words_predict = test_line[1]

                    test_context_str = ''
                    test_predict_str = ''

                    next_word_history = None
                    pointer_history = None
                    for word in test_context:
                        test_context_str = test_context_str + ' ' + word
                        #fix: lookup word if not in corpus
                        context_word_idx = corpus.dictionary.word2idx[word]
                        input.data.fill_(context_word_idx)
                        output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)

                        rnn_out = rnn_outs[-1].squeeze(0)
                        output_flat = output.view(-1, ntokens)
                        next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
                        pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])


                    st.write ('Test Context: ' + test_context_str)
                    # ready for inference, last context word has been processed and hidden state & cache updated

                    for i in range(num_words_predict):

                        # start predicting
                        softmax_output_flat = torch.nn.functional.softmax(output_flat)
                        theta = theta_val
                        lambdah = lambdasm_val
                        vocab_loss = softmax_output_flat[0]
                        p = vocab_loss

                        logits = torch.mv(pointer_history, rnn_out.data.view(-1,1).squeeze())
                        ptr_attn = torch.nn.functional.softmax(theta * Variable(logits)).view(-1, 1)
                        ptr_dist = (ptr_attn.expand_as(next_word_history) * next_word_history).sum(0).squeeze()
                        p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss
                        word_idx = torch.multinomial(p.data, 1)[0]

                        input.data.fill_(word_idx)
                        word = corpus.dictionary.idx2word[word_idx]

                        test_preds.append(word)
                        test_predict_str = test_predict_str + ' ' + word

                        # update state with predicted word
                        output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)
                        rnn_out = rnn_outs[-1].squeeze(0)
                        output_flat = output.view(-1, ntokens)
                        next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
                        pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])

                    st.write ('Test Preds: ' + test_predict_str)

                    references = [full_context]
                    candidate = [test_context + test_preds]

                    st.write('*** Debug ***')
                    st.write(references, len(references))
                    st.write(candidate, len(candidate))
                    
                    st.write()

                    new_BLEU_score = corpus_bleu(references, candidate)
                    st.write(new_BLEU_score)

                    st.write('*** end ***')
                    BLEU_score = BLEU_score + new_BLEU_score

                st.write('Total BLEU: ', BLEU_score)



                    #calc BLEU score

# #context routine
# context_str = ''
# prediction_str = ''
# if len (args.context) > 0:
#     print('** processing current context **')
#     next_word_history = None
#     pointer_history = None
#     context_words = args.context.split()
#     for word in context_words:
#         context_str = context_str + ' ' + word
#         #fix: lookup word if not in corpus
#         context_word_idx = corpus.dictionary.word2idx[word]
#         input.data.fill_(context_word_idx)
#         output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)
#         if args.window > 0:
#             #while stepping thorugh context, record words output + hidden states
#             rnn_out = rnn_outs[-1].squeeze(0)
#             output_flat = output.view(-1, ntokens)
#             next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
#             pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])
#         word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
#         word_idx = torch.multinomial(word_weights, 1)[0]
#         word2 = corpus.dictionary.idx2word[word_idx]
#         prediction_str = prediction_str + ' ' + word2
#     st.write ('Context: ' + context_str)
#     st.write ('Prediction: ' + prediction_str)


# pred_str = ''
# pred_str2 = ''
# print('** start prediction with pointer **')
# #with open(args.outf, 'w') as outf:
# for i in range(args.words):
#     output, hidden, rnn_outs, _ = model(input, hidden, return_h=True)
#     if args.window > 0:
#         #while stepping thorugh preds, record words + hidden states
#         rnn_out = rnn_outs[-1].squeeze(0)
#         output_flat = output.view(-1, ntokens)
#         next_word_history = one_hot(context_word_idx, ntokens) if next_word_history is None else torch.cat([next_word_history, one_hot(context_word_idx, ntokens)])
#         pointer_history = rnn_out.data if pointer_history is None else torch.cat([pointer_history, rnn_out.data])
#         softmax_output_flat = torch.nn.functional.softmax(output_flat)
#         theta = args.theta
#         lambdah = args.lambdasm
#         vocab_loss = softmax_output_flat[0]
#         p = vocab_loss
#         #     if start_idx + idx > window:
#         valid_next_word = next_word_history
#         valid_pointer_history = pointer_history
#         logits = torch.mv(valid_pointer_history, rnn_out.data.view(-1,1).squeeze())
#         ptr_attn = torch.nn.functional.softmax(theta * Variable(logits)).view(-1, 1)
#         ptr_dist = (ptr_attn.expand_as(valid_next_word) * valid_next_word).sum(0).squeeze()
#         p = lambdah * ptr_dist + (1 - lambdah) * vocab_loss
#         #     ###
#         #     target_loss = p[targets[idx].data]
#         #     loss += (-torch.log(target_loss)).data[0]
#         # with pointer
#         #word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
#         word_idx = torch.multinomial(p.data, 1)[0]
#
#         #without pointer_history
#         word_weights_no_ptr = output.squeeze().data.div(args.temperature).exp().cpu()
#         word_idx_no_ptr = torch.multinomial(vocab_loss.data, 1)[0]
#
#         #word_idx = torch.multinomial(word_weights, 1)[0]
#         input.data.fill_(word_idx)
#         word = corpus.dictionary.idx2word[word_idx]
#         word_no = corpus.dictionary.idx2word[word_idx_no_ptr]
#
#         pred_str = pred_str + ' ' + word
#         pred_str2 = pred_str2 + ' ' + word_no
#
# print ('Pointer: ' + pred_str)
# print ('No Pointer: ' + pred_str2)
