import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

class GaussPlotter:

    def __init__(self, cov):
        self.cov = cov
        self.lam, self.U = np.linalg.eig(cov)
    
    def plot(self, num_sample, x0, y0, x1, y1, delta):
        samples = np.random.normal(0, 1.0, (len(self.lam), num_sample))
        samples *= self.lam.reshape(-1, 1)
        samples *= 1 / delta
        samples = np.ceil(samples)
        num_x = int((x1 - x0)/delta)
        num_y = int((y1 - y0)/delta)
        X = np.arange(int(x0/delta), int(x1/delta))
        Y = np.arange(int(y0/delta), int(y1/delta))
        x, y = np.meshgrid(X, Y)
        xy = x.flatten() * 100000000 + y.flatten()
        samples = samples[0] * 100000000 + samples[1]
        samples = np.sum(samples.reshape(1, -1) == xy.reshape(-1, 1), axis=1)
        samples = samples.reshape(num_x, num_y)
        samples = samples / num_sample

        xya = np.stack([x, y], axis=-1)
        xy = xya.reshape((-1, 2)) * delta
        e = - 0.5 * np.sum(np.dot(xy, np.linalg.inv(self.cov)) * xy, axis=1)
        norm = np.linalg.norm(self.cov)
        p = 1 / ((2 * np.pi) ** (len(self.lam) / 2)) * 1 / norm * np.exp(e)
        p = p.reshape(num_x, num_y)


        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
        xa, ya = np.hsplit(np.dot(xya, self.U.T), 2)
        ax.plot_surface(xa.reshape(num_x, num_y), ya.reshape(num_x, num_y), samples.transpose(0, 1) / (delta ** 2))
        ax.plot_surface(x, y, p)
        plt.show()
        pass

    def sample(self,):
        pass

    def plot_target(self,):
        pass

def main():
    cov = np.array([[2, 1],
                    [1, 1]])
    gp = GaussPlotter(cov)
    gp.plot(10000, -5, -5, 5, 5, 0.1)

if __name__ == "__main__":
    main()