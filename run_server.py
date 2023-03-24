import os
os.environ["EVENTLET_NO_GREENDNS"] = 'yes'

import eventlet
eventlet.monkey_patch(socket=True, select=True)

import os
os.environ["EVENTLET_NO_GREENDNS"] = 'yes'

import tqdm
from time import strftime
import logging

import sklearn
import sklearn.neighbors
import sklearn.tree
import sklearn.metrics._pairwise_distances_reduction._datasets_pair
import sklearn.metrics._pairwise_distances_reduction._middle_term_computer
import sklearn.metrics._pairwise_distances_reduction

import unidic_lite
from unidic_lite import unidic

try:
    import torch
except:
    pass

from gandy.app import run_server

run_server()

# TODO: Better importing. Eventlet is wonky with Pyinstaller.