from os import listdir, getenv
from os.path import isfile, join
import re
import numpy 
import argparse
import logging 
import theano
import embeddings
import theano.tensor as T

import convolve
from convolve import *
learning_rate = 0.8
batch_size = 500
layer2_size = 50
window_size = 3

(iw,wi,vs) = embeddings.load_encoding()
datasets = load_datasets(wi,vs,window_size=window_size)
