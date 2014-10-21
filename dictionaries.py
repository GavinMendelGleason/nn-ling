#!/usr/bin/env python

"""Code implementing dictionary construction for bag of words model"""


class Dictionary(object): 
    def __init__(self,dictionary_size=10000): 
        self.dictionary = {}
        self.reverse = {}
        self.counts = {}
        self.dictionary_size = dictionary_size
        self.initialised = False

    def register(self,word): 
        if not word in self.counts: 
            self.counts[word] = 1            
        else:
            self.counts[word] += 1
    
    def encode(self,word):
        if self.initialised and word.lower() in self.dictionary: 
            return self.dictionary[word.lower()]
        elif not self.initialised:
            raise Exception("Dictionary not initialised!")
        else: 
            return None
        
    def decode(self,code): 
        if self.initialised and code in self.reverse:
            return self.reverse[code]
        elif not self.initialised: 
            raise Exception("Dictionary not initialised!")
        else: 
            return None
    
    def initialise(self,cutoff=10000): 
        current_cull_size = 0
        size = len(self.counts)
        # first cull the words with too low count
        for i in range(0,size-self.dictionary_size): 
            found = False
            while not found: 
                for key in self.counts:
                    count = self.counts[key]
                    if count == current_cull_size: 
                        self.counts.pop(key)
                        found = True
                if not found:
                    current_cull_size +=1
        i = 0 
        for key in self.counts: 
            self.dictionary[key] = i
            self.reverse[i] = key
            i+=1
        self.initialised = True
        
    def vectorise(self,word): 
        arr = numpy.zeros(self.dictionary_size)
        res = self.encode(word) 
        if res: 
            arr[res] = 1
        return arr

    def tensorise(self,document): 
        encoded_words = []
        for word in document: 
            encoded_words.append(self.vectorise(word))
        return numpy.matrix(encoded_words)

def tokenise(document): 
    words = []
    stack = ''
    for char in document: 
        if char == r'.' or char == '?' or char == '!': 
            if not stack == '':
                words.append(stack)                
                stack = ''
            words.append(char)
        elif char == r' ' or char == '\n': 
            if not stack == '':
                words.append(stack)
                stack = ''
        else: 
            stack += char

    if stack != '': 
        words.append(stack)
    return words

def register(documents): 
    
    d = Dictionary()

    for doc in documents: 
        words = tokenise(doc)
        print words
        for word in words:
            d.register(word.lower())

    d.initialise() 
    return d
