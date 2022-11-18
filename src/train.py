#!/usr/bin/python3

# load command line args
import argparse
parser = argparse.ArgumentParser()
parser_io = parser.add_argument_group('input/output')
parser_io.add_argument('--model_path')
parser_io.add_argument('--data_path', default='data/cap_1.section1')
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

# initialize pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())
    
def train():
    
    # load the text file
    data = open(args.data_path, 'r').read()
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:"\'?*!#_-[]{}() \n'
    data_size, vocab_size = len(data), len(chars)

    logging.info("Data has {} characters, {} unique".format(data_size, vocab_size))
    
    # char to index and index to char maps
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    # convert data from chars to indices
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]
    
    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    logger.debug(f'data.shape={data.shape}')
    
    # model instance
    rnn = RNN(vocab_size, vocab_size, args.hidden_size, args.num_layers).to(device)
    
    # load checkpoint if True
    if args.load_checkpoint:
        rnn.load_state_dict(torch.load(args.model_path))
        print("Model loaded successfully !!")
        print("----------------------------------------")
    
    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
    
    # training loop
    for i_epoch in range(1, args.epochs+1):
        
        # random starting point (1st 100 chars) from data to begin
        data_ptr = np.random.randint(100)
        #data_ptr = 0
        n = 0
        running_loss = 0
        hidden_state = None
        
        while True:
            input_seq = data[data_ptr : data_ptr+args.seq_len]
            target_seq = data[data_ptr+1 : data_ptr+args.seq_len+1]
            
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # compute loss
            loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
            running_loss += loss.item()
            
            # compute gradients and take optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update the data pointer
            #while data[data_ptr] != '\n' and data_ptr < len(data)-1:
                #data_ptr += 1
            data_ptr += args.seq_len
            n +=1
            
            # if at end of data : break
            if data_ptr + args.seq_len + 1 > data_size:
                break
            
        # print loss and save weights after every epoch
        print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
        torch.save(rnn.state_dict(), args.model_path)
        
        # sample / generate a text sequence after every epoch
        data_ptr = 0
        hidden_state = None
        
        # random character from data to begin
        rand_index = np.random.randint(data_size-1)
        input_seq = data[rand_index : rand_index+1]
        
        print("----------------------------------------")
        while True:
            # forward pass
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # construct categorical distribution and sample a character
            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()
            
            # print the sampled character
            print(ix_to_char[index.item()], end='')
            
            # next input is current output
            input_seq[0][0] = index.item()
            data_ptr += 1
            
            if data_ptr > args.output_seq_len:
                break
            
        print("\n----------------------------------------")
        
if __name__ == '__main__':
    train()


