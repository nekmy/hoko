import numpy as np
from scipy.stats import multivariate_normal
from PIL import Image
import cv2
import matplotlib.pylab as plt
import pygraph

from kmean import KMean

class GrabCutter:

    def __init__(self, num_k):
        self.num_k = num_k
        self.mus = np.random.rand(2, self.num_k)
        dim = 3
        covs = np.random.randn(self.num_k, dim, dim)
        covs = np.abs(covs)
        self.covs = np.stack([np.dot(cov, cov.T) for cov in covs], axis=0)
        self.pis = np.random.rand(num_k)
        self.pis = self.pis / np.sum(self.pis)

    def cut(self, image, num_iter):
        self.image = image
        self.alpha = np.ones_like(image[:, :, 0])
        self.k = np.zeros_like(self.alpha)
        for a in (0, 1):
            indices = np.where(self.alpha==a)
            target = image[indices]
            kmean = KMean(self.num_k, target)
            kmean.loop_update(num_iter=20)
            k, _ = kmean.get_result()
            self.k[indices] = k
        
        for _ in range(num_iter):
            e_step()
            
        
        
        
    def e_step(self):
        data = []
        for a in (0, 1):
            data.append([multivariate_normal.pdf(image, self.mus[a, k], self.covs[a, k]) for k in range(self.num_k)])
        data = np.array(data)
        data = -np.log(data) - np.log(self.pis)
        self.k = np.argmax(data[self.alpha], axis=1)
        
        for a in (0, 1):
            for k in range(self.num_k):
                indices = np.where((self.alpha==a)and(self.k==k))
                self.mus[a, k] = np.mean(self.image[indices], axis=0)
                self.covs[a, k] = np.cov(self.image[indices])
                self.pis[a, k] = np.sum(np.where((self.alpha==a)and(self.k==k))) / np.sum(np.where((self.alpha==a))

        


        



def main():
    cutter = GrabCutter(num_k=5)
    image = np.array(Image.open("image/pig.jpg"))
    cutter.cut(image, 1000)

if __name__ == "__main__":
    main()