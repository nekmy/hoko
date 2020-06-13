import numpy as np
import matplotlib.pylab as plt

class GibbsSampler:

    def __init__(self):
        self.samples = []

    def sample(self, num_iter, a=0, x0=0, y0=0):
        x = x0
        y = y0
        for _ in range(num_iter):
            x = np.random.normal(a*y, 1)
            y = np.random.normal(a*x, 1)
            self.samples.append([x, y])

    def plot_samples(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        samples = np.array(self.samples)
        ax.scatter(samples[:, 0], samples[:, 1])
        plt.show()

def main():
    sampler = GibbsSampler()
    sampler.sample(10000, a=1.0)
    sampler.plot_samples()

if __name__ == "__main__":
    main()