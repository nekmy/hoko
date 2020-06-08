import numpy as np
import matplotlib.pylab as plt

# from bolzman import BoltzmanMachine
from gbrbmsigma2 import BoltzmanMachine
from loadmnist import load_mnist

def noise(image, noise_rate):
    plane = np.ones_like(image)
    plane = plane.flatten()
    inv_indices = np.random.choice(len(plane), int(len(plane)*noise_rate), replace=False)
    plane[inv_indices] = 0
    plane = plane.reshape(image.shape)
    return (image == plane).astype(np.int)

def main():
    num_node = 28 * 28
    bm = BoltzmanMachine(num_node, num_hidden=100, lr=0.001)
    images = load_mnist(normalize=False)["train_images"][:10]
    # images = (images > 0.5).astype(np.int)
    num_iter = 1000
    for i in range(num_iter):
        target = np.random.choice(len(images))
        x_sampled = images[target]
        bm.update(x_sampled)
    #noised = noise(images, 0.1)
    noised = images.copy()
    
    fig = plt.figure()
    H = np.floor(np.sqrt(len(noised)))
    W = 2 * np.ceil(np.sqrt(len(noised)))
    for i in range(len(noised)):
        ax = fig.add_subplot(H, W, 2*i+1)
        ax.imshow(noised[i].reshape(28, 28))
        mean = bm.gibss_xhx(noised[i], 1000)
        ax = fig.add_subplot(H, W, 2*i+2)
        ax.imshow(mean.reshape(28, 28))
    plt.show()
    
if __name__ == "__main__":
    main()