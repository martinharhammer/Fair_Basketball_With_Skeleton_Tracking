import os, pickle

def save_stub(stub_path, obj):
    if stub_path is None:
        return
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(obj, f)

def read_stub(read_from_stub, stub_path):
    if read_from_stub and stub_path and os.path.exists(stub_path):
        with open(stub_path, "rb") as f:
            return pickle.load(f)
    return None

