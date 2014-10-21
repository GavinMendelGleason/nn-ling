#!/usr/bin/env python
import os
from os import listdir, getenv
from os.path import isfile, join
import sys
import re
import numpy 
import argparse
import logging 
import theano
import embeddings
import theano.tensor as T
import time 

def onlyfiles(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

class Windows(object):
    def __init__(self,document,left_pad='<s>',right_pad='</s>',window_size=3): 
        self.left_pad = left_pad
        self.right_pad = right_pad
        self.window_size = window_size
        self.document = document
        self.current = -(self.window_size / 2)
        self.doc_len = len(document)
        
    def __iter__(self): 
        return self
        
    def __next__(self): 
        """ For python 3 """
        return self.next()

    def next(self): 
        if self.current >= (self.doc_len - (self.window_size / 2)):
            raise StopIteration()
        else:
            left_slice = [self.left_pad] * max(0,-self.current)
            right_slice = [self.right_pad] * max(0,self.current + self.window_size - self.doc_len)
            start = max(0,self.current)
            end = min(self.doc_len,self.current+self.window_size)
            result = left_slice + list(self.document[start:end]) + right_slice
            self.current += 1
            return result

def wordToNum(wi,word): 
    if word in wi: 
        return wi[word] 
    else: 
        return 0

def encode_sentence(x,wi,vs,window_size=3):
    w = Windows(x,window_size=window_size)
    total = []
    for window in w:
        current = []
        for word in window:
            current += list(vs[wordToNum(wi,word)])
        total.append(current)
    return total

def read_labelled_sentences(wi,vs,f, window_size=3):
    inputFile = open(f, 'rb')

    sentences = []
    sentence = []
    labels = []
    for line in inputFile:
        if line == '\n': 
            continue
        (word,l) = line.lower().rstrip('\n').split()
        sentence.append(word)
        if l == 'person':
            label = 1
        else:
            label = 0
        labels.append(label)

        if word == ".":
            sentences.append(sentence) 
            sentence = []
    # in case we didn't end with a sentence...
    if sentence !=[]:
        sentences.append(sentence)

    total = []
    for sentence in sentences: 
        total += encode_sentence(sentence,wi,vs, window_size=window_size)
                
    return (total,labels)

def load_datasets(wi,vs,window_size=3,basedir=None): 
    if not basedir: 
        basedir = os.getcwd() + '/pa4-ner/data' 
    
    (train_arr_x,train_arr_y) = read_labelled_sentences(wi,vs,basedir+'train', window_size=window_size)
    # load as theano shared arrays
    train_x = theano.shared(numpy.asarray(train_arr_x,
                                    dtype=theano.config.floatX),
                      borrow=True)
    train_y = theano.shared(numpy.asarray(train_arr_y, 
                                    dtype='int32'),
                      borrow=True)

    (dev_arr_x,dev_arr_y) = read_labelled_sentences(wi,vs,basedir+'dev', window_size=window_size)
    test_x = theano.shared(numpy.asarray(dev_arr_x[0:len(dev_arr_x)/2], 
                                         dtype=theano.config.floatX),
                           borrow=True)
    test_y = theano.shared(numpy.asarray(dev_arr_y[0:len(dev_arr_y)/2], 
                                         dtype='int32'),
                           borrow=True)

    valid_x = theano.shared(numpy.asarray(dev_arr_x[len(dev_arr_x)/2:], 
                                         dtype=theano.config.floatX),
                           borrow=True)
    valid_y = theano.shared(numpy.asarray(dev_arr_y[len(dev_arr_y)/2:], 
                                         dtype='int32'),
                           borrow=True)

    return {'train' : (train_x,train_y),
            'test' : (test_x,test_y),
            'valid' : (valid_x,valid_y)}
        

__LOG_FORMAT__ = "%(asctime)-15s %(message)s"
__LIB_PATH__ = getenv("HOME") + '/lib/nn-ling/'
__DEFAULT_PATH__ = __LIB_PATH__ + 'corpus/'
__LOG_PATH__ = __LIB_PATH__ + 'nn-ling.log'
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Playing with word embeddings.')
    #parser.add_argument('--corpus-dir', help='Corpus directory', default=__DEFAULT_PATH__)
    parser.add_argument('--log', help='Log file', default=__LOG_PATH__)
    parser.add_argument('--learning-rate', help='Learning rate', default="0.001")
    parser.add_argument('--batch-size', help='Batch size', default="100")
    parser.add_argument('--hidden-layer1-size',help='Size of the first hidden layer', default = "150")
    parser.add_argument('--hidden-layer2-size',help='Size of the second hidden layer', default = "100")
    parser.add_argument('--window-size',help='Size of the convolution window [must be odd]', default = "5")
    parser.add_argument('--n-train-batches', help='Number of training batches to process', default= "10")
    parser.add_argument('--n-epochs', help='Number of epochs', default = "100")
    parser.add_argument('--label-dim', help='Dimensionality of the label space', default="2")
    parser.add_argument('--l2-weight', help='Weight for L2', default="0.3")
    parser.add_argument('--l1-weight', help='Weight for L1', default="0.3")
    args = vars(parser.parse_args())

    learning_rate = float(args['learning_rate'])
    batch_size = int(args['batch_size'])
    n_epochs = int(args['n_epochs'])
    label_dim = int(args['label_dim'])

    layer2_size = int(args['hidden_layer1_size'])
    layer3_size = int(args['hidden_layer2_size'])
    window_size = int(args['window_size'])

    l1_weight = float(args['l1_weight'])
    l2_weight = float(args['l2_weight'])

    # set up logging
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(filename=args['log'],level=logging.INFO,
                        format=__LOG_FORMAT__)

    (iw,wi,vs) = embeddings.load_encoding()

    # Debugging
    # theano.config.compute_test_value = 'warn'
    
    datasets = load_datasets(wi,vs,window_size=window_size)
    train_x,train_y = datasets['train']
    valid_x,valid_y = datasets['valid']
    test_x,test_y = datasets['test']

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

    #random numbers 
    rng = numpy.random.RandomState(123)

    # empirical 
    (_,layer1_size) = theano.function([],T.shape(train_x))()
    
    index = T.lscalar()
    # Debugging info
    # theano.config.compute_test_value = 'warn'

    x = T.matrix('x',dtype=theano.config.floatX)
    # provide Theano with a default test-value
    x.tag.test_value = numpy.random.rand(batch_size, 50 * window_size)
    
    y = T.ivector('y')
    y.tag.test_value = numpy.random.rand(batch_size)

    # Randomly initialise hidden layers
    W_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (layer1_size + layer2_size)),
        high=numpy.sqrt(6. / (layer1_size + layer2_size)),
        size=(layer1_size, layer2_size)), dtype=theano.config.floatX)

    b1_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / layer2_size), 
        high=numpy.sqrt(6. / layer2_size),
        size=(layer2_size,)), dtype=theano.config.floatX)
     
    U_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (layer2_size + layer3_size)),
        high=numpy.sqrt(6. / (layer2_size + layer3_size)),
        size=(layer2_size,layer3_size)), dtype=theano.config.floatX)

    b2_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / layer3_size), 
        high=numpy.sqrt(6. / layer3_size),
        size=(layer3_size,)), dtype=theano.config.floatX)
    
    Z_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / (layer3_size + label_dim)),
        high=numpy.sqrt(6. / (layer3_size + label_dim)),
        size=(layer3_size,label_dim)), dtype=theano.config.floatX)

    b3_values = numpy.asarray(rng.uniform(
        low=-numpy.sqrt(6. / label_dim), 
        high=numpy.sqrt(6. / label_dim),
        size=(label_dim,)), dtype=theano.config.floatX)
    
    b1 = theano.shared(value=b1_values, name='b1')
    W = theano.shared(value=W_values, name='W')
    
    b2 = theano.shared(value=b2_values, name='b2')
    U = theano.shared(value=U_values, name='U')

    b3 = theano.shared(value=b3_values, name='b3')
    Z = theano.shared(value=Z_values, name='Z')
    
    hidden_layer_1 = T.tanh(T.dot(x,W) + b1)
    hidden_layer_2 = T.tanh(T.dot(hidden_layer_1,U) + b2)
    h = T.nnet.sigmoid(T.dot(hidden_layer_2, Z) + b3)
    
    negative_log_likelihood = T.mean(T.dot(y,T.log(h)) - T.dot(1 - y, T.log(1 - h)))
    y_pred=T.argmax(h, axis=1)

    # L1 ; force absolute value of weights to be small
    L1 = (
        abs(W).mean()
        + abs(U).mean()
        + abs(Z).mean()
    )
    # square of L2 norm ; force square of L2 norm to be small
    L2_sqr = (
        (W ** 2).mean()
        + (U ** 2).mean()
        + (Z ** 2).mean()
    )
    
    cost = l2_weight * L2_sqr + l1_weight * L1 + negative_log_likelihood

    errors = T.mean(T.neq(y_pred, y))

    g_W = T.grad(cost=cost, wrt=W)
    g_b1 = T.grad(cost=cost, wrt=b1)
    g_U = T.grad(cost=cost, wrt=U) 
    g_b2 = T.grad(cost=cost, wrt=b2) 
    g_Z = T.grad(cost=cost, wrt=Z) 
    g_b3 = T.grad(cost=cost, wrt=b3) 

    updates = [(W, W - learning_rate * g_W),
               (b1, b1 - learning_rate * g_b1), 
               (U, U - learning_rate * g_U), 
               (b2, b2 - learning_rate * g_b2), 
               (Z, Z - learning_rate * g_Z), 
               (b3, b3 - learning_rate * g_b3)]

    
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={
                                      x: train_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_y[index * batch_size:(index + 1) * batch_size]})

    test_model = theano.function(inputs=[index],
                                 outputs=errors,
                                 givens={
                                     x: test_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
                                     outputs=errors,
                                     givens={
                                         x: valid_x[index * batch_size: (index + 1) * batch_size],
                                         y: valid_y[index * batch_size: (index + 1) * batch_size]})
        
    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            #print "Minibatch: %s" % minibatch_index
            minibatch_avg_cost = train_model(minibatch_index)
            #print "Minibatch avg cost: %s" % minibatch_avg_cost
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print "Validation: %s" % validation_frequency
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


    ## Here we should dump best parameters.
    # f = open(path,'wb')
    # W.pickle(f)
