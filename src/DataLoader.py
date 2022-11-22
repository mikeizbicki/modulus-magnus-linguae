# initialize logging
import logging
logger = logging.getLogger(__name__)

# imports
from collections import Counter
import itertools
import random
import string


class DataLoader():

    def __init__(self, language, paths, translations=['en']):
        self.language = language
        self.paths = paths
        logger.debug(f'paths={paths}')

        # load the data
        lines = []
        for path in paths:
            with open(path, 'rt', encoding='utf-8') as fin:
                lines_la = fin.readlines()
                lines_path = [{'la': line_la} for line_la in lines_la]
            try:
                with open(path+'.en', encoding='utf-8') as fin:
                    lines_en = fin.readlines()
                    assert(len(lines_en) == len(lines_la))
                    for line, line_en in zip(lines_path, lines_en):
                        line['en'] = line_en
            except FileNotFoundError:
                pass
            lines.extend(lines_path)

        # compute the data
        self.data = [line['la'] for line in lines]
        logger.info(f'len(self.data)={len(self.data)}')

        self.tokens = self.language.tokenize(' '.join(self.data))
        logger.info(f'len(self.tokens)={len(self.tokens)}')

        # gather statistics
        tokens_counter = Counter(self.tokens)
        logger.debug(f'tokens_counter.most_common(10)={tokens_counter.most_common(10)}')

    def replace_characters(self, text, max_replace=3, num_samples=3, prefix=''):
        samples = []
        indexes = list(range(len(text)))
        for i in range(num_samples):
            random_indexes = []
            choices = list(range(1,max_replace+1))
            for i in range(random.choice(choices)):
                random_indexes.append(random.choice(indexes))
            random_indexes = list(sorted(set(random_indexes)))
            sample = prefix + 'c|'
            last_random_index = -1
            for random_index in random_indexes:
                sample += text[last_random_index+1:random_index] + '~'
                last_random_index = random_index
            sample += text[last_random_index+1:]
            sample += '|'
            for random_index in random_indexes:
                sample += text[random_index]
            samples.append(sample)
        return list(set(samples))

    def replace_words(self, text, min_word_size=2, max_replace=2, num_samples=3, prefix=''):
        samples = []
        tokens = self.language.tokenize(text)
        tokens_indexes = list(range(len(tokens)))
        valid_tokens_indexes = []
        for tokens_index in tokens_indexes:
            token = tokens[tokens_index]
            if len(token) >= min_word_size:
                valid_tokens_indexes.append(tokens_index)
        if valid_tokens_indexes == []:
            return []
        for i in range(num_samples):
            random_indexes = []
            choices = list(range(1,max_replace+1))
            for i in range(random.choice(choices)):
                token_index = random.choice(valid_tokens_indexes)
                random_indexes.append(token_index)
            random_indexes = list(sorted(set(random_indexes)))
            last_random_index = -1
            new_tokens = []
            for random_index in random_indexes:
                new_tokens.extend(tokens[last_random_index+1:random_index]) 
                new_tokens.append(' ~')
                last_random_index = random_index
            new_tokens.extend(tokens[last_random_index+1:])
            sample = prefix + 'w|'
            sample += self.language.detokenize(new_tokens)
            sample += '|'
            sample += ' '.join([tokens[random_index] for random_index in random_indexes])
            samples.append(sample)
        return list(set(samples))

    def replace_punctuation(self, text, max_replace=1, num_samples=3, prefix=''):
        samples = []
        tokens = self.language.tokenize(text)
        tokens_indexes = list(range(len(tokens)))
        valid_tokens_indexes = []
        for tokens_index in tokens_indexes:
            token = tokens[tokens_index]
            if len(token) == 1 and token in string.punctuation:
                valid_tokens_indexes.append(tokens_index)
        for i in range(num_samples):
            random_indexes = []
            choices = list(range(1,max_replace+1))
            for i in range(random.choice(choices)):
                token_index = random.choice(valid_tokens_indexes)
                random_indexes.append(token_index)
            random_indexes = list(sorted(set(random_indexes)))
            last_random_index = -1
            new_tokens = []
            for random_index in random_indexes:
                new_tokens.extend(tokens[last_random_index+1:random_index]) 
                new_tokens.append('~')
                last_random_index = random_index
            new_tokens.extend(tokens[last_random_index+1:])
            sample = prefix + 'p|'
            sample += self.language.detokenize(new_tokens)
            sample += '|'
            sample += ' '.join([tokens[random_index] for random_index in random_indexes])
            samples.append(sample)
        return list(set(samples))

    def replace_suffix(self, text, min_word_size=3, max_replace=5, max_len=3, min_len=1, num_samples=3, prefix=''):
        samples = []
        tokens = self.language.tokenize(text)
        tokens_indexes = list(range(len(tokens)))
        valid_tokens_indexes = []
        for tokens_index in tokens_indexes:
            token = tokens[tokens_index]
            if len(token) >= min_word_size:
                valid_tokens_indexes.append(tokens_index)
        if valid_tokens_indexes == []:
            return []
        for i in range(num_samples):
            random_indexes = []
            suffix_lengths = []
            choices = list(range(1,max_replace+1))
            for i in range(random.choice(choices)):
                token_index = random.choice(valid_tokens_indexes)
                random_indexes.append(token_index)
                suffix_lengths.append(random.randrange(min_len, max_len+1))
            random_indexes = list(sorted(set(random_indexes)))
            last_random_index = -1
            new_tokens = []
            for random_index_i,random_index in enumerate(random_indexes):
                new_tokens.extend(tokens[last_random_index+1:random_index]) 
                new_tokens.append(tokens[random_index][:-suffix_lengths[random_index_i]] + '~')
                last_random_index = random_index
            new_tokens.extend(tokens[last_random_index+1:])
            sample = prefix + 's|'
            sample += self.language.detokenize(new_tokens)
            sample += '|'
            sample += ' '.join([tokens[random_index][-suffix_lengths[random_index_i]:] for random_index_i,random_index in enumerate(random_indexes)])
            samples.append(sample)
        return list(set(samples))


    def line_to_samples(self, line):
        line = line.strip()
        if len(line) == 0:
            return []
        if line[0] == '#':
            return []
        samples = []

        # generate 'l|' samples
        samples.append('la|'+line+'|')
        samples.extend(self.replace_characters(line, prefix='l'))
        samples.extend(self.replace_suffix(line, prefix='l'))
        samples.extend(self.replace_words(line, prefix='l', num_samples=10, max_replace=4))
        samples.extend(self.replace_punctuation(line, prefix='l'))

        return samples
     
    def __iter__(self):
        lines = self.data #self.data.split('\n')
        samples = map(lambda line: self.line_to_samples(line), lines)
        iter1 = itertools.chain.from_iterable(samples)

        line_pairs = ['ll|' +l1.strip() + '|' + l2 for l1,l2 in zip(lines[:-1], lines[1:])]
        return iter(itertools.chain(iter1, line_pairs))

class Batchify():
    def __init__(self, data_loader, batch_size):
        self.data_loader = data_loader
        self.batch_size = batch_size

    def __iter__(self):
        itr = iter(self.data_loader)
        return BatchifyIterator(itr, self.batch_size)

class BatchifyIterator():
    def __init__(self, itr, batch_size):
        self.itr = itr
        self.batch_size = batch_size

    def __next__(self):
        batch = []
        for i in range(self.batch_size):
            try:
                batch.append(next(self.itr))
            except StopIteration:
                break
        if batch == []:
            raise StopIteration
        return batch


def shuffle(generator, buffer_size):
    while True:
        buffer = list(itertools.islice(generator, buffer_size))
        if len(buffer) == 0:
            break
        random.shuffle(buffer)
        for item in buffer:
            yield item

def scramble(gen, buffer_size):
    buf = []
    i = iter(gen)
    while True:
        try:
            e = next(i)
            buf.append(e)
            if len(buf) >= buffer_size:
                choice = random.randint(0, len(buf)-1)
                buf[-1],buf[choice] = buf[choice],buf[-1]
                yield buf.pop()
        except StopIteration:
            random.shuffle(buf)
            yield from buf
            return
