#!/usr/bin/env python
import scipy

def read_vocab(f):
    """Returns two dictionaries (int_word,word_int), one mapping integers 
       to words, the other the reverse mapping"""
    int_word = {}
    word_int = {}
    i = 0
    for line in f: 
        word_int[line] = i
        int_word[i] = line
        i += 1
    f.close()

    return (int_word,word_int)

def read_vectors(f): 
    """Read in the 50 dimensional embeddigs"""
    vectors = {}
    i = 0
    for line in f:
        vectors[i] scipy.array([float(x) for x in line.split(" ")])
        i += 1
    f.close() 
    return vectors


def read_data(basedir="/home/rowan/Documents/code/python/convolution/pa4-ner/data/"): 
    vocab = open(basedir+'vocab.txt', 'rb')
    int_word,word_int = read_vocab(vocab)
    wordVectors = open(basedir+'wordVectors.txt',  'rb') 
    vectors = read_vectors(wordVectors) 
    return (int_word,word_int,vectors)
        
