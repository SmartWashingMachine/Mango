from onnxruntime import (
    InferenceSession,
)

def session_has_cuda(sess: InferenceSession):
    providers = sess.get_providers()
    for prov in providers:
        if 'Dml' in prov:
            return True

    return False

def get_from_buffer(name: str, buffers, shapes):
    return buffers[name].view(shapes[name]).clone()

DEVICE_ID = 0 # device ID. TODO: Use other than cuda:0
DEVICE_TO_USE = 'cpu' #f'cuda:{DEVICE_ID}'