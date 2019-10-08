#!/usr/bin/python

import argparse
import glob
import re
import operator


### arguments ###
### --recog_dir: the directory where the recognition files from inferency.py are placed
### --ground_truth_dir: the directory where the framelevel ground truth can be found
parser = argparse.ArgumentParser()
parser.add_argument('--obs_perc', default='30')
parser.add_argument('--obs_dir', default='obs')
parser.add_argument('--decode_dir', default='results')
args = parser.parse_args()
filelist = glob.glob(args.decode_dir + '/*')

print('Evaluate %d video files...' % len(filelist))
# loop over all recognition files and evaluate the frame error
for filename in filelist:
    print(filename.split('/')[1])
    appo = filename.split('/')[1]
    observed = []
    with open(filename, 'r') as f:
      recognized = f.read().split('\n')[5].split() # framelevel recognition is in 6-th line of file
      f.close()
    for i in range(int(len(recognized) * int(args.obs_perc)/ 100)):
            observed.append(recognized[i])
    with open(args.obs_dir+'/'+appo,'w') as writeFile:
        for o in observed:
            writeFile.writelines(o)
            writeFile.write('\n')


