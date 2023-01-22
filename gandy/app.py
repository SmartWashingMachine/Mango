__version__ = '0.1.0'

# Eventlet must monkey patch BEFORE the other imports.
import eventlet
eventlet.monkey_patch()

from time import strftime
import logging
from flask import Flask, request
from gandy.full_pipelines.advanced_pipeline import AdvancedPipeline
from flask_socketio import SocketIO
import os
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)

# ONNX seems to be funky with asynchronous logick magick. With the default ping timeout (5000ms), it's likely that the client will drop the connection midway through the process.
# Of course, a long ping timeout is not ideal either.
#socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, ping_timeout=100000)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=100000)

# disable for debug
translate_pipeline = AdvancedPipeline()

@app.route('/ping')
def ping_route():
    return 'Hello World!', 200

logger = logging.getLogger('Gandy')
logger.setLevel(logging.DEBUG)

save_folder_path = os.path.expanduser('~/Documents/Mango/logs')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path, exist_ok=True)

fh = logging.FileHandler(f'{save_folder_path}/backend_logs_{strftime("%d_%m_%Y")}.txt', encoding='utf-8')

fh_f = logging.Formatter(' %(name)s :: %(asctime)s.%(msecs)03d :: %(levelname)s :: %(message)s', '%d %H:%M:%S')
fh.setFormatter(fh_f)

fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

logger.info('Running app.')
print('Running app.')

def run_server():
    if True: # for randy ONLY TODO
        socketio.run(app, host='0.0.0.0')
    else:
        socketio.run(app)

import gandy.book_routes
import gandy.paraphrase_routes
import gandy.task1_routes
import gandy.task2_routes
import gandy.task3_routes
import gandy.config_routes
import gandy.randy_routes
