import numpy as np

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

        self.bias_x = self.bias_x + self.lr * (x_sampled - self.x_takeovered)
        self.bias_h = self.bias_h + self.lr * (h_sampled - self.h_takeovered)
        self.weight_xh = self.weight_xh + self.lr *\
            (x_sampled.reshape(-1, 1) * h_sampled.reshape(1, -1) -\
                self.x_takeovered.reshape(-1, 1) * self.h_takeovered.reshape(1, -1))

    def prob_x(self, h):
        energy = self.bias_x + np.dot(self.weight_xh, h.reshape(-1, 1)).flatten()
        return prob_func(energy)

    def prob_h(self, x):
        energy = self.bias_h + np.dot(x.reshape(1, -1), self.weight_xh).flatten()
        return prob_func(energy)
    
    def sample_x_value(self, h):
        threshold = self.prob_x(h)
        x = np.random.rand(self.num_visible) < threshold
        x = x.astype(np.float64)
        return x
    
    def sample_h_value(self, x):
        threshold = self.prob_h(x)
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
        x_mean = np.mean(x_gbs, axis=0)

        return x_mean
    
    def gibss_xhx(self, x_sampled, num_sample):
        x_gbs = []
        for _ in range(num_sample):
            h = self.sample_h_value(x_sampled)
            x = self.sample_x_value(h)
            x_gbs.append(x)
        x_gbs = np.stack(x_gbs)
        x_mean = np.mean(x_gbs, axis=0)

        return x_mean

def main():
    num_node = 8
    waltz = RandomWaltz(num_node)
    pattern1 = np.array([0, 0, 0, 0, 0, 1, 0, 1])
    pattern2 = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    pattern3 = np.array([0, 1, 0, 1, 1, 0, 0, 0])
    pattern4 = np.array([0, 0, 0, 1, 0, 1, 1, 0])
    waltz.add_pattern(pattern1)
    waltz.add_pattern(pattern2)
    waltz.add_pattern(pattern3)
    waltz.add_pattern(pattern4)
    boltzman_machine = BoltzmanMachine(num_node, num_hidden=100, lr=0.01)
    num_iter = 10000
    for _ in range(num_iter):
        pattern_id = np.random.choice(waltz.num_pattern)
        x_sampled = waltz.sample(pattern_id, 1)[0]
        boltzman_machine.update(x_sampled)
    
    target_pattern = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    voids_mask = np.array([False, False, False, False, True, True, True, True])
    x_mean = boltzman_machine.gibss_smaple(target_pattern, voids_mask, num_iter=1000)
    print(x_mean)

if __name__ == "__main__":
    main()

    