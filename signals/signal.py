import math

import numpy as np
# import tensorflow as tf
import matplotlib.pylab as plt
from matplotlib import patches

class World:

    def __init__(self, time_interval):
        self.time_interval = time_interval
        self.objects = []
    
    def append(self, obj):
        if type(obj) in (tuple, list):
            self.objects += obj
        else:
            self.objects.append(obj)
    
    def draw(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_ylim(-5, 5)
        ax.set_xlim(-5, 5)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        elems = []
        while True:
            self.step(ax, elems)
            plt.pause(0.001)
    
    def step(self, ax, elems):
        while elems:
            elems.pop().remove()
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "step"):
                obj.step(self.time_interval)
            


class Object:

    def __init__(self, pose, radius, color=(0, 0, 0)):
        self.pose = pose
        self.radius = radius
        self.color = color
    
    def step(self):
        pass
    
    def draw(self, ax, elems):
        pass

class Golem(Object):
    
    def __init__(self, pose, radius, color=(0, 0, 0, 1), agent=None):
        super().__init__(pose, radius, color)
        self.agent = agent
    
    def step(self, time, state=False):
        # nu: 速度
        # omega: 角速度
        # time: 時間間隔
        if self.agent:
            nu, omega = self.agent.decide(state)
            self.pose = self.state_transision(self.pose, nu, omega, time)
    
    def draw(self, ax, elems):
        x, y, theta = self.pose
        x_n = x + self.radius * np.cos(theta)
        y_n = y + self.radius * np.sin(theta)
        elems += ax.plot([x, x_n], [y, y_n], color=self.color)
        circle = patches.Circle([x, y], radius=self.radius, color=self.color, fill=False)
        ax.add_patch(circle)
        elems.append(circle)
    
    @classmethod
    def state_transision(cls, pose, nu, omega, time):
        x, y, theta = pose
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(theta),
                                    nu*math.sin(theta),
                                    omega]) * time
        else:
            return pose + np.array([nu/omega*(math.sin(theta+omega*time)-math.sin(theta)),
                                    nu/omega*(-math.cos(theta+omega*time)+math.cos(theta)),
                                    omega*time])

class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def step(self):
        pass

    def decide(self, state):
        return self.nu, self.omega

def main():
    world = World(0.1)
    agent1 = Agent(1.0, 50/180*math.pi)
    init_pose1 = (0, 0, 0)
    golem1 = Golem(init_pose1, radius=0.2, color=(0.1, 0.8, 0.8, 1), agent=agent1)
    world.append(golem1)
    world.draw()

if __name__ == "__main__":
    main()