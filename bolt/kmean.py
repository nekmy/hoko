import numpy as np
import matplotlib.pylab as plt
import cv2

class KMean:

    def __init__(self, num_class, samples):
        self.num_class = num_class
        self.mu = np.random.rand(num_class, samples.shape[1])
        self.responsivilities = np.zeros((len(samples), num_class))
        self.samples = samples
    
    def loop_update(self, num_iter):
        for _ in range(num_iter):
            self.e_step()
            self.m_step()
    
    def get_result(self):
        k = np.where(self.responsivilities==True)[1]
        return k, self.mu

    def e_step(self):
        samples = np.expand_dims(self.samples, axis=1)
        mu = np.expand_dims(self.mu, axis=0)
        distanse = np.linalg.norm(samples - mu, axis=2)
        min_k = np.argmin(distanse, axis=1)
        self.responsivilities = np.identity(self.num_class)[min_k]
    
    def m_step(self):
        samples = np.expand_dims(self.samples, axis=1)
        responsivilities = np.expand_dims(self.responsivilities, axis=2)
        samples = samples * responsivilities
        summed = np.sum(samples, axis=0)
        num_r = np.sum(self.responsivilities, axis=0)
        num_r = np.expand_dims(num_r, axis=1)
        indices = np.where(num_r > 0)[0]
        mean = summed[indices] / num_r[indices]
        self.mu[indices] = mean
        indices = np.where(num_r == 0)[0]
        if len(indices):
            self.mu[indices] = np.random.rand(len(indices), samples.shape[2])

    
    def scatter(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap("tab10")
        k = np.where(self.responsivilities==True)[1]

        for i in range(self.num_class):
            indices = np.where(k==i)
            samples_k = self.samples[indices]
            ax.scatter(samples_k[:, 0], samples_k[:, 1], c=cmap(i))
        
        plt.show()
    
    def restore(self):
        k = np.where(self.responsivilities==True)[1]
        return self.mu[k]

class Sampler:

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
    sampler = Sampler(locs, covs, responsivility)
    num_sample = 10000
    num_class = 3
    samples = sampler.sample(num_sample)

    img_path = "image/pig.jpg"
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    samples = img.reshape(-1, 3)
    samples = samples / 255.0
    num_class = 10
    k_mean = KMean(num_class, samples)
    num_iter = 20
    k_mean.loop_update(num_iter)
    restored = k_mean.restore()
    img = restored.reshape(h, w, -1) * 255
    img = img.astype(np.uint8)
    cv2.imshow("cat", img)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    main()