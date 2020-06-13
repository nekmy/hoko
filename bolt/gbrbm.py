import numpy as np
import matplotlib.pylab as plt

from em import MultiNormalSampler

def prob_func(x):
    return np.exp(x) / (1 + np.exp(x))

class RandomWaltz:
    def __init__(self, num_node):
        self.num_node = num_node
        self.patterns = []

    def add_pattern(self, pattern):
        # 1をとる確率の配列
        assert len(pattern) == self.num_node
        self.patterns.append(pattern)
    
    def sample(self, pattern_id, num_sample):
        threshold = np.array(self.patterns[pattern_id])
        sample = np.random.rand(num_sample, self.num_node) < threshold
        return sample.astype(np.float64)
    
    @property
    def num_pattern(self):
        return len(self.patterns)

class BoltzmanMachine:
    def __init__(self, num_visible, num_hidden, lr):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = lr
        self.bias_x = np.random.randn(num_visible) * 0.01
        self.bias_h = np.random.randn(num_hidden) * 0.01
        self.weight_xh = np.random.randn(num_visible, num_hidden) * 0.01
        self.sigma = np.ones(num_visible)
        
        self.x_takeovered = None
        self.h_takeovered = None

    def update(self, x_sampled):
        """
        近似したい確率分布から取得したサンプルを入力として
        パラメータを更新
        """
        # hを取得
        h_sampled = self.sample_h_value(x_sampled)

        if self.x_takeovered is None:
            self.h_takeovered = h_sampled
        self.x_takeovered = self.sample_x_value(self.h_takeovered)
        self.h_takeovered = self.sample_h_value(self.x_takeovered)

        # update
        self.bias_x = self.bias_x + self.lr * (x_sampled - self.x_takeovered) / (self.sigma ** 2)
        self.bias_h = self.bias_h + self.lr * (h_sampled - self.h_takeovered)
        self.weight_xh = self.weight_xh + self.lr *\
            (x_sampled.reshape(-1, 1) * h_sampled.reshape(1, -1) -\
                self.x_takeovered.reshape(-1, 1) * self.h_takeovered.reshape(1, -1)) / (self.sigma**2).reshape(-1, 1)
        
        z = np.log(self.sigma**2)
        dz = np.exp(-z * ((0.5 * (x_sampled - self.bias_x) ** 2 - x_sampled * np.dot(self.weight_xh, h_sampled.reshape(-1, 1)).flatten())\
            - (0.5 * (self.x_takeovered - self.bias_x) ** 2 - self.x_takeovered * np.dot(self.weight_xh, self.h_takeovered.reshape(-1, 1)).flatten())))
        z = z + dz
        #self.sigma = np.exp(z) ** 0.5

    def prob_h_1(self, x):
        energy = self.bias_h + np.dot((x / self.sigma**2).reshape(1, -1), self.weight_xh).flatten()
        return prob_func(energy)
    
    def sample_x_value(self, h):
        loc = self.bias_x + self.sigma * np.dot(self.weight_xh, h.reshape(-1, 1)).flatten()
        loc = loc.flatten()
        x = np.random.normal(loc, self.sigma)
        return x
    
    def sample_x_value_mult(self, h, num_x):
        loc = self.bias_x + self.sigma * np.dot(self.weight_xh, h.reshape(-1, 1)).flatten()
        loc = loc.flatten()
        x = np.random.normal(loc, self.sigma, (num_x, self.num_visible))
        return x
    
    def sample_h_value(self, x):
        threshold = self.prob_h_1(x)
        h = np.random.rand(self.num_hidden) < threshold
        h = h.astype(np.float64)
        return h
    
    def gibss_smaple(self, x_sampled, voids_mask, num_iter, buffer_ratio=0.1):
        x_gbs = []
        x = x_sampled
        num_buffer = int(num_iter * buffer_ratio)
        for _ in range(num_buffer + num_iter):
            h = self.sample_h_value(x)
            x = self.sample_x_value(h)
            x = x * voids_mask + x_sampled * (voids_mask == False)
            x_gbs.append(x)
        x_gbs = np.stack(x_gbs)[num_buffer:]

        return x_gbs

    def gibss_smaple_mean(self, x_sampled, voids_mask, num_iter, buffer_ratio=0.1):
        x_gbs = self.gibss_smaple(x_sampled, voids_mask, num_iter, buffer_ratio)
        x_mean = np.mean(x_gbs, axis=0)

        return x_mean
    
    def x_sample(self, h, num_sample):
        x = self.sample_x_value_mult(h, num_sample)
        return x

def main():
    num_node = 2
    boltzman_machine = BoltzmanMachine(num_node, num_hidden=10, lr=0.01)
    num_iter = 10000
    locs = np.array([[5.0, 5.0],
                     [-5.0, 5.0],
                     [5.0, -5.0]])
    covs = np.array([[[0.5, 0.0],
                      [0.0, 0.5]],
                     [[0.1, 1],
                      [1, 0.1]],
                     [[0.1, 2],
                      [2, 0.1]]])
    responsivilities = np.array([0.0, 0.5, 0.5])
    sampler = MultiNormalSampler(locs, covs, responsivilities)
    
    num_sample = 100000
    samples = sampler.sample(num_sample)
    #samples = samples - np.mean(samples, axis=0)
    sigma = np.mean(samples**2, axis=0)**0.5
    #samples = samples / sigma
    samples += np.array([1, 1])
    for _ in range(num_iter):
        index = np.random.choice(num_sample)
        sample = samples[index]
        boltzman_machine.update(sample)
    
    target_pattern = np.array([0, 0])
    voids_mask = np.array([True, True])
    x_gbs = boltzman_machine.gibss_smaple(target_pattern, voids_mask, num_iter=1000)
    #x = boltzman_machine.x_sample(np.array([1, 0, 0, 0]), 100)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.scatter(*samples.T)
    ax = fig.add_subplot(224)
    ax.scatter(*x_gbs.T)
    plt.show()

if __name__ == "__main__":
    main()

    