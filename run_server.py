import os
os.environ["EVENTLET_NO_GREENDNS"] = 'yes'

import eventlet
eventlet.monkey_patch(socket=True, select=True)

import os
os.environ["EVENTLET_NO_GREENDNS"] = 'yes'

import tqdm
from time import strftime
import logging

from gandy.app import run_server

run_server()

#it works?1 but how do we put it in...