import os
from urllib import request
import gzip

import numpy as np

def load_labels(file_path):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_images(file_path, normalize=False):
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28*28)
        if normalize:
            data = data.astype(np.float)
            data = data - np.mean(data, axis=0, keepdims=True)
            data = data / np.sqrt(np.mean(data**2, axis=0, keepdims=True))
    return data


def load_mnist(normalize=False):
    dir = "mnist"
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = {"test_images": "t10k-images-idx3-ubyte.gz",
                  "test_labels": "t10k-labels-idx1-ubyte.gz",
                  "train_images": "train-images-idx3-ubyte.gz",
                  "train_labels": "train-labels-idx1-ubyte.gz"}

    for v in file_names.values():
        url = url_base + "/" + v
        file_path = dir + "/" + v
        if not os.path.exists(file_path):
            request.urlretrieve(url, file_path)
    
    mnist = {}

    mnist["test_images"] = load_images(dir+"/"+file_names["test_images"], normalize)
    mnist["test_labels"] = load_labels(dir+"/"+file_names["test_labels"])
    mnist["train_images"] = load_images(dir+"/"+file_names["train_images"], normalize)
    mnist["train_labels"] = load_labels(dir+"/"+file_names["train_labels"])
    
    return mnist


def main():
    mnist = load_mnist()
    pass

if __name__ == "__main__":
    main()