from onnxruntime import (
    InferenceSession,
)

def session_has_cuda(sess: InferenceSession):
    providers = sess.get_providers()

    for prov in providers:
        if 'CUDA' in prov:
            return True

    return False

def get_from_buffer(name: str, buffers, shapes):
    if name not in buffers or buffers[name] is None:
        return None
    return buffers[name].view(shapes[name])#.clone()

DEVICE_ID = 0 # device ID. TODO: Use other than cuda:0
DEVICE_TO_USE = f'cuda:{DEVICE_ID}' #'cpu' #f'cuda:{DEVICE_ID}'
