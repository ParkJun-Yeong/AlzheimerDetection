"""
[Preparations]
- Settings for compatability with python 2.x.
- import requires.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


"""
[Load and Preprocess Data]
- Reformat data file.
- Load the data into structures that we can work with.
"""
# Print some datafile to see the original format.
corpus_name = 'pitt_corpus_cookie_theft'
corpus = os.path.join("data", corpus_name)

def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

printLines(os.path.join(corpus.movie_lines.txt))

# Create formatted data file. (query and answer)
def loadLines(fileName)