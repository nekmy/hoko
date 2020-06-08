import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt
from matplotlib import cm
import cv2

class EMMultiNormal:

    def __init__(self, num_class, samples):
        num_samples, dim = samples.shape
        self.num_class = num_class
        self.samples = samples
        self.locs = np.random.randn(num_class, dim)
        covs = np.random.randn(num_class, dim, dim)
        covs = np.abs(covs)
        self.covs = np.stack([np.dot(cov, cov.T) for cov in covs], axis=0)
        self.mixpi = np.random.randn(num_class)
        self.mixpi = self.mixpi / np.sum(self.mixpi)
        self.responsivilities = np.zeros((num_samples, num_class))
        color_indices = np.arange(num_class) / num_class
        self.colors = cm.jet(color_indices)
    
    def loop_update(self, num_iter):
        for _ in range(num_iter):
            self.e_step()
            self.m_step()
    
    def e_step(self):
        
        responsivilities = []
        
        for i in range(self.num_class):
            responsivilities.append(multivariate_normal.pdf(self.samples, self.locs[i], self.covs[i]))
        responsivilities = np.stack(responsivilities, axis=1)
        responsivilities = responsivilities * np.expand_dims(self.mixpi, axis=0)
        responsivilities = responsivilities / np.sum(responsivilities, axis=1, keepdims=True)
        self.responsivilities = responsivilities
    
    def m_step(self):
        num_k = np.sum(self.responsivilities, axis=0)
        for k in range(self.num_class):
            summed = self.responsivilities[:, k].reshape(-1, 1) * self.samples
            summed = np.sum(summed, axis=0)
            self.locs[k] = summed / num_k[k]
            delta = self.samples - self.locs[k].reshape(1, -1)
            delta = np.expand_dims(delta, axis=2) * np.expand_dims(delta, axis=1)
            summed = self.responsivilities[:, k].reshape(-1, 1, 1) * delta
            summed = np.sum(summed, axis=0)
            self.covs[k] = summed / num_k[k]
            self.mixpi[k] = num_k[k] / len(self.samples)
    
    def scatter(self):
        color = np.expand_dims(self.responsivilities, axis=2) * np.expand_dims(self.colors, axis=0)
        color = np.sum(color, axis=1)
        color = np.where(color<=1, color, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(*self.samples.T, c=color)
        plt.show()

class MultiNormalSampler:

    def __init__(self, locs, covs, responsivilities):
        assert len(locs) == len(covs) == len(responsivilities)
        self.locs = locs
        self.covs = covs
        
        self.responsivilities = responsivilities
        
    
    def sample(self, num_sample):
        samples = []
        for loc, cov, r in zip(self.locs, self.covs, self.responsivilities):
            n = int(num_sample*r)
            sample = np.random.multivariate_normal(loc, cov, n)
            samples.append(sample)
        samples = np.concatenate(samples)
        return samples

def main():
    locs = np.array([[1.0, 1.0],
                     [-1.0, -0.5],
                     [-3.0, 2.0]])
    covs = np.array([[[0.5, 0.0],
                      [0.0, 0.5]],
                     [[0.3, 0.5],
                      [0.5, 0.7]],
                     [[0.1, -0.5],
                      [-0.5, 0.3]]])
    responsivility = np.array([0.2, 0.3, 0.5])
    sampler = MultiNormalSampler(locs, covs, responsivility)
    num_sample = 10000
    num_class = 3
    samples = sampler.sample(num_sample)

    em_multi_normal = EMMultiNormal(3, samples)
    em_multi_normal.loop_update(100)
    em_multi_normal.scatter()


    

if __name__ == "__main__":
    main()