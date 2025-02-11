#!/usr/bin/python2.7

import numpy as np
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--decoded_path", default="data/split1.test")
parser.add_argument("--result_path", default="result1/")


args, unknown = parser.parse_known_args()



### helper function for parallelized Viterbi decoding ##########################
def decode(queue, log_probs, decoder, index2label):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            with open('results/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except queue.Empty:
            pass


### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
with open('data/mapping.txt', 'r') as f:
    content = f.read().split('\n')[0:-1]
    for line in content:
        label2index[line.split()[1]] = int(line.split()[0])
        index2label[int(line.split()[0])] = line.split()[1]

### read test data #############################################################
with open(args.decoded_path, 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset('data', video_list, label2index, shuffle = False)

# load prior, length model, grammar, and network
load_iteration = 10000
log_prior = np.log( np.loadtxt(args.result_path+'prior.iter-' + str(load_iteration) + '.txt') )
grammar = PathGrammar(args.result_path+'grammar.txt', label2index)
length_model = PoissonModel(args.result_path+'lengths.iter-' + str(load_iteration) + '.txt', max_length = 2000)
forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
forwarder.load_model(args.result_path+'network.iter-' + str(load_iteration) + '.net')

# parallelization
n_threads = 4

# Viterbi decoder
viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
# forward each video
log_probs = dict()
queue = mp.Queue()
for i, data in enumerate(dataset):
    sequence, _ = data
    video = list(dataset.features.keys())[i]
    queue.put(video)
    log_probs[video] = forwarder.forward(sequence) - log_prior
    log_probs[video] = log_probs[video] - np.max(log_probs[video])
# Viterbi decoding
procs = []
for i in range(n_threads):
    p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
    procs.append(p)
    p.start()
for p in procs:
    p.join()

