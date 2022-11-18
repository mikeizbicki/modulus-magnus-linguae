#!/usr/bin/python3

# load command line args
import argparse
parser = argparse.ArgumentParser()
parser_io = parser.add_argument_group('input/output')
parser_io.add_argument('--model_path')
parser_io.add_argument('--data_path', default='data/cap_1.section1')
parser_io.add_argument('--capitvlvm', type=int)
parser_io.add_argument('--data_dir', default='data')
parser_architecture = parser.add_argument_group('architecture')
parser_architecture.add_argument('--hidden_size', type=int, default=512)
parser_architecture.add_argument('--seq_len', type=int, default=100)
parser_architecture.add_argument('--num_layers', type=int, default=3)
parser_optimizer = parser.add_argument_group('optimizer')
parser_optimizer.add_argument('--lr', type=float, default=0.002)
parser_optimizer.add_argument('--epochs', type=int, default=100)
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
import glob
import re
from collections import Counter

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
        return self.generate_samples(**kwargs)[0][1]

        
    def generate_samples(
            rnn,
            prompt='\n\n',
            max_length=100,
            sentence_break=False,
            paragraph_break=True,
            temperature=0.8,
            beams=3,
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
        return list(zip(logprobs, texts))


class Latin():
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:"\'?*!#_-[]{}() \n'

    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    def text_to_tensor(self, text):
        data = []
        for i, ch in enumerate(text):
            data.append(self.char_to_ix[ch])
        data = torch.tensor(data)
        data = torch.unsqueeze(data, dim=1)
        return data

    def texts_to_tensor(self, texts):
        tensors = [self.text_to_tensor(text) for text in texts]
        return torch.cat(tensors, dim=1)

language = Latin()


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


class DataLoader():

    def __init__(self, paths):
        # load the data
        file_contents = []
        for path in paths:
            with open(path, 'rt', encoding='utf-8') as fin:
                file_contents.append(fin.read())
        data = '\n\n'.join(file_contents)
        self.data = data

        # gather statistics
        logger.info(f'len(data)={len(data)}')
        grammar_tokens = data.split()
        logger.debug(f'len(grammar_tokens)={len(grammar_tokens)}')
        tokens = re.findall(r"\w+|[^\w\s]", data, re.UNICODE)
        logger.info(f'len(tokens)={len(tokens)}')
        tokens_counter = Counter(tokens)
        logger.debug(f'tokens_counter.most_common(10)={tokens_counter.most_common(10)}')
        #logger.debug(f'tokens_counter.most_common()[-10:]={tokens_counter.most_common()[-10:]}')
     
        #tokens_lower = re.findall(r"\w+|[^\w\s]", data.lower(), re.UNICODE)
        #logger.info(f'len(tokens_lower)={len(tokens_lower)}')
        #tokens_lower_counter = Counter(tokens_lower)
        #logger.debug(f'tokens_lower_counter.most_common(10)={tokens_lower_counter.most_common(10)}')
        #logger.debug(f'tokens_lower_counter.most_common()[-10:]={tokens_lower_counter.most_common()[-10:]}')

    def __iter__(self):
        return DataIterator(self.data)

class DataIterator():
    def __init__(self, data):
        self.paragraphs = data.split('\n\n')
        self.i = 0

    def __next__(self):
        if self.i >= len(self.paragraphs):
            raise StopIteration
        paragraph = self.paragraphs[self.i]
        self.i += 1
        return paragraph.strip()


def train():

    # load training data
    paths = []
    for capitvlvm in range(1, args.capitvlvm+1):
        globpath = os.path.join(args.data_dir, 'capitvlvm_'+str(capitvlvm)+'.section*')
        paths.extend(list(glob.glob(globpath)))

    data_loader = DataLoader(paths)

    # model instance
    rnn = RNN(language, args.hidden_size, args.num_layers).to(device)
    
    # load checkpoint if True
    if args.load_checkpoint:
        logging.info('loading model')
        rnn.load_state_dict(torch.load(args.model_path))
        logging.info('loaded model')
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
    
    # training loop
    for epoch_i in range(1, args.epochs+1):
        running_loss = 0
        
        for sample_i,sample in enumerate(data_loader):
            # convert data from chars to indices
            data = language.text_to_tensor(sample)
            data = data.to(device)
            #logger.debug(f'data.shape={data.shape}')
            hidden_state = None
    
            input_seq = data[:-1]
            target_seq = data[1:]
            
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(epoch_i, running_loss/sample_i))

        print("----------------------------------------")
        print(rnn.generate_sample()) 
        print("\n----------------------------------------")
    torch.save(rnn.state_dict(), args.model_path)
        
if __name__ == '__main__':
    #eval_pensvmC()
    train()


