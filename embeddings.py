#!/usr/bin/env python
import numpy
import os 

def read_vocab(f):
    """Returns two dictionaries (int_word,word_int), one mapping integers 
       to words, the other the reverse mapping"""
    int_word = {}
    word_int = {}
    i = 0
    for line in f:
        l = line.rstrip('\n')
        word_int[l] = i
        int_word[i] = l
        i += 1
    f.close()

    return (int_word,word_int)

def read_vectors(f): 
    """Read in the 50 dimensional embeddings"""
    vectors = {}
    i = 0
    for line in f:
        vectors[i] = [float(x) for x in line.split(" ")[0:50]]
        i += 1
    f.close() 
    return vectors

def read_labels(f):
    """Read in word, label pairs"""
    labels = []
    for line in f: 
        (word,ner_class) = line.rstrip('\n').split(' ')
        labels[word] = ner_class
    return labels

def load_encoding(basedir=None):
    if not basedir: 
        basedir = os.getcwd() + '/pa4-ner/data' 
    vocab = open(basedir+'vocab.txt', 'rb')
    int_word,word_int = read_vocab(vocab)
    wordVectors = open(basedir+'wordVectors.txt',  'rb') 
    vectors = read_vectors(wordVectors) 

    return (int_word,word_int,vectors)
        
