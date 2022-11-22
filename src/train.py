#!/usr/bin/python3

# load command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=['train', 'pensvmC', 'data'], default='train')
parser_io = parser.add_argument_group('input/output')
parser_io.add_argument('--model_path')
parser_io.add_argument('--data_path', default='data/cap_1.section1')
parser_io.add_argument('--capitvlvm', type=int, required=True)
parser_io.add_argument('--data_dir', default='data')
parser_io.add_argument('--save_every', type=int, default=20)
parser_architecture = parser.add_argument_group('architecture')
parser_architecture.add_argument('--hidden_size', type=int, default=512)
parser_architecture.add_argument('--seq_len', type=int, default=100)
parser_architecture.add_argument('--num_layers', type=int, default=3)
parser_optimizer = parser.add_argument_group('optimizer')
parser_optimizer.add_argument('--lr', type=float, default=0.002)
parser_optimizer.add_argument('--epochs', type=int, default=100)
parser_optimizer.add_argument('--batch_size', type=int, default=10)
parser_optimizer.add_argument('--load_checkpoint', action='store_true')
parser_debug = parser.add_argument_group('debug')
parser_debug.add_argument('--output_seq_len', type=int, default=200)
args = parser.parse_args()

# initialize logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ensure args.model_path is sane
import os
if not args.model_path:
    args.model_path = 'models/test'
dirname = os.path.dirname(args.model_path)
os.makedirs(dirname, exist_ok=True)

# imports
import string
import re
import glob

from DataLoader import DataLoader, Batchify, shuffle, scramble

# initialize pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, language, hidden_size, num_layers):
        super(RNN, self).__init__()
        input_size = len(language.chars)
        output_size = len(language.chars)
        self.language = language
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())


    def generate_sample(self, **kwargs):
        samples, logprobs = self.generate_samples(**kwargs)
        return samples[0]

        
    def generate_samples(
            rnn,
            prompt='',
            max_length=100,
            sentence_break=False,
            paragraph_break=True,
            temperature=1, #0.8,
            beams=4,
            samples_per_beam=2,
            ):
      with torch.no_grad():
        device = next(rnn.parameters()).device

        texts = [prompt[0]]*beams
        logprobs = torch.tensor([0.0]*beams)
        input_seq = language.texts_to_tensor(texts).to(device)
        hidden_state = None
        for pos_i in range(len(prompt)+max_length):
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)

            # next input is current output
            if pos_i < len(prompt)-1:
                next_char = prompt[pos_i+1]
                next_ix = language.char_to_ix[next_char]
                for beam_i in range(beams):
                    input_seq[0][beam_i] = next_ix
                    texts[beam_i] += next_char
            
            # construct categorical distribution and sample a character
            else:
                allprobs = F.softmax(torch.squeeze(temperature*output, dim=0), dim=1)
                dist = Categorical(allprobs)
                new_texts = []
                new_logprobs = []
                for sample_i in range(samples_per_beam):
                    index = dist.sample()
                    for beam_i,text in enumerate(texts):
                        next_ix = index[beam_i].item()
                        next_char = language.ix_to_char[next_ix]
                        input_seq[0][beam_i] = next_ix
                        new_texts.append(texts[beam_i] + next_char)
                        new_logprobs.append(logprobs[beam_i] + torch.log(allprobs[beam_i][index[beam_i]]))
                    
                new_texts, new_logprobs = zip(*sorted(zip(new_texts, new_logprobs)))

                texts = [new_texts[0]]
                logprobs = [new_logprobs[0]]
                for i,new_text in enumerate(new_texts):
                    if texts[-1] != new_text:
                        texts.append(new_text)
                        logprobs.append(new_logprobs[i])
                    if len(texts) >= beams:
                        break

            # early stop 
            #if pos_i > len(prompt):
                #if paragraph_break and len(text) > 2 and text[-1] == '\n' and text[-2] == '\n':
                    #break
                #if sentence_break and text[-1] in '.!?':
                    #break

        #texts = [''.join(text) for text in texts]
        #for i,text in enumerate(texts):
            #print(f'{logprobs[i]:0.4f} {text}')
            #print("texts=",texts)
        texts = [text.strip() for text in texts]
        return texts, logprobs


class Latin():
    chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:"\'?*!#_-[]{}() \n|~'

    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    def text_to_tensor(self, text, min_size=0, max_size=256):
        data = []
        padding_length = min_size - len(text)
        text += ' '*padding_length
        for i, ch in enumerate(text[:max_size]):
            data.append(self.char_to_ix[ch])
        data = torch.tensor(data)
        data = torch.unsqueeze(data, dim=1)
        return data

    def texts_to_tensor(self, texts, max_size=256):
        min_size = max([len(text) for text in texts])
        tensors = [self.text_to_tensor(text, min_size=min_size) for text in texts]
        return torch.cat(tensors, dim=1)

    def tokenize(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return tokens

    def detokenize(self, tokens):
        newtokens = []
        for i, token in enumerate(tokens):
            if i == 0 or token[0] in string.punctuation or token[0] == ' ':
                newtokens.append(token)
            else:
                newtokens.append(' ' + token)
        return ''.join(newtokens)

language = Latin()


def eval_pensvm(rnn, filepath, verbose=False, max_tests=None):
    # load the test cases
    lines = open(filepath, 'r').readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line)>0 and line[0] != '#']

    # compute scores
    scores = {'topk': 0, 'top1': 0, 'tests': 0}
    for line in lines:
        sections = line.split('|')
        prompt = '|'.join(sections[:2])+'|'
        samples, logprobs = rnn.generate_samples(prompt=prompt)
        if verbose:
            print(f'true=<<{line}>>')
            print(f'pred=<<{samples[0]}>>')
            print('---')
        if samples[0] == line:
            scores['top1'] += 1
        if line in samples:
            scores['topk'] += 1
        scores['tests'] += 1
        if max_tests and scores['tests'] >= max_tests:
            break
    return scores


def eval_pensvmC():
    # load the text file
    data = open('data/cap_1.pensvmC', 'r').read()
    prompts = data.split('\n')

    data_size, vocab_size = len(data), len(language.chars)
    logging.info("Data has {} characters, {} unique".format(data_size, vocab_size))

    # model instance
    rnn = RNN(language, args.hidden_size, args.num_layers).to(device)
    
    # load checkpoint if True
    if args.load_checkpoint:
        logging.info('loading model')
        rnn.load_state_dict(torch.load(args.model_path))
        logging.info('loaded model')
    
    # evaluate
    for prompt in prompts:

        # skip bad prompts
        prompt = prompt.strip()
        if len(prompt) == 0 or prompt[0] == '#':
            continue

        # print question/answer
        print('--------------------')
        text = rnn.generate_sample(prompt=prompt, sentence_break=True)
        #text = text.replace('\n', ' ')
        print(text)


def train():

    # load training data
    paths = []
    for capitvlvm in range(1, args.capitvlvm+1):
        globpath = os.path.join(args.data_dir, 'capitvlvm_'+str(capitvlvm)+'.section?')
        paths.extend(list(glob.glob(globpath)))

    # model instance
    rnn = RNN(language, args.hidden_size, args.num_layers).to(device)
    
    # load checkpoint if True
    if args.load_checkpoint:
        logging.info('loading model')
        rnn.load_state_dict(torch.load(args.model_path))
        logging.debug('loaded model')
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
    
    data_loader = DataLoader(language, paths)

    # training loop
    for epoch_i in range(1, args.epochs+1):
        running_loss = 0

        data = scramble(data_loader, buffer_size=64*args.batch_size)
        data = Batchify(data, batch_size=args.batch_size)

        for sample_i,batch in enumerate(data):
            # convert data from chars to indices
            data = language.texts_to_tensor(batch)
            data = data.to(device)
            hidden_state = None
    
            input_seq = data[:-1]
            target_seq = data[1:]
            
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)

            
            # compute loss
            output_ = torch.flatten(output, 0, 1)
            target_seq_ = torch.flatten(target_seq, 0, 1)
            loss = loss_fn(output_, target_seq_)
            running_loss += loss.item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch_i, running_loss/sample_i))

        print("----------------------------------------")
        #print(rnn.generate_sample(prompt='lw|Sicilia ~ est.|'))
        #print(rnn.generate_sample(prompt='lw|Italia insula ~ est.|'))
        #print(rnn.generate_sample(prompt='lw|Rhenus ~ est.|'))
        #print(rnn.generate_sample(prompt='lw|Brundisium ~ est.|'))
        max_tests = 3
        scores = eval_pensvm(rnn, 'data/capitvlvm_1.pensvmA', verbose=True, max_tests=max_tests)
        scores = eval_pensvm(rnn, 'data/capitvlvm_1.pensvmB', verbose=True, max_tests=max_tests)
        scores = eval_pensvm(rnn, 'data/capitvlvm_1.pensvmC', verbose=True, max_tests=max_tests)
        #print("scores=",scores)
        print("\n----------------------------------------")

        if (epoch_i+1)%args.save_every == 0:
            logging.info('saving model weights')
            torch.save(rnn.state_dict(), args.model_path)
            logging.debug('saved model weights')
        

if __name__ == '__main__':
    if args.action == 'train':
        train()

    elif args.action == 'pensvmC':
        eval_pensvmC()

    elif args.action == 'data':
        paths = []
        for capitvlvm in range(1, args.capitvlvm+1):
            globpath = os.path.join(args.data_dir, 'capitvlvm_'+str(capitvlvm)+'.section*')
            paths.extend(list(glob.glob(globpath)))

        data_loader = DataLoader(language, paths)
        #data_loader = scramble(data_loader, buffer_size=128)
        #data_loader = Batchify(data_loader, batch_size=3)
        for i, data in enumerate(data_loader):
            print("data=",data)
            #print()
            if i > 10:
                break



