import sys
import random
import re

def corpus_generator(file, fileout, min_words=50, max_words=150, corpus_size=50000):
    corpus = []
    regex = re.compile('[@_!#$%^&*()<>«»?/\|}{~:]') 
    for line in open(file):
        line = line.strip()
        seq_len = len(line)
        
        if seq_len >= min_words and seq_len <= max_words:
            if bool(regex.search(line)):
                if random.random() < .1:
                    corpus.append(line)
            else:
                corpus.append(line)
        
        if len(corpus) >= corpus_size:
            break
    with open(fileout, 'w') as f:
        for line in corpus:
            f.write(line + '\n')
    return corpus



if __name__ == '__main__':
    file, fileout = sys.argv[1], sys.argv[2]
    corpus_generator(file, fileout)
    