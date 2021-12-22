import numpy as np
import matplotlib.pyplot as plt

class Plotting:
    def __init__(self, name, xlim=[-5,5], ylim=[-5,5], is_grid=True):
        self.name = name
        self.xlim = xlim
        self.ylim = ylim
        self.is_grid = is_grid

        self.ax = plt.axes()
        self.ax.set_title(name)
        self.ax.grid(is_grid)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.axis('equal')
    
    def plot_path(self, path, label):
        path = np.array(path)
        self.ax.plot(path[:,0], path[:,1], label=label)
        plt.legend()

    def plot_animation(self, path1, path2, ref1, ref2, radius=0.25):
        path1 = np.array(path1)
        path2 = np.array(path2)
        ref1 = np.array(ref1)
        ref2 = np.array(ref2)

        length = len(path1)
        for i in range(length):
            plt.clf()
            # Reference
            plt.plot(ref1[:i,0], ref1[:i,1], "-b")
            plt.plot(ref2[:i,0], ref2[:i,1], "-b")

            plt.plot(path1[:i,0], path1[:i,1], "-g", label="Omni 1")
            self.draw_circle(path1[i,:2], radius, 'g')

            plt.plot(path2[:i,0], path2[:i,1], "-r", label="Omni 2")
            self.draw_circle(path2[i,:2], radius, 'r')

            
            plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event:
                                            [exit(0) if event.key == 'escape' else None])
            plt.title("Omni1: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}\nOmni2: vx: {:.2f}, vy: {:.2f}, omega: {:.2f}".format(\
                path1[i,3], path1[i,4], path1[i,5], path2[i,3], path2[i,4], path2[i,5]))
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.grid(self.is_grid)
            plt.legend()
            plt.pause(0.001)

    def draw_circle(self, center, radius, color):
        q = np.arange(0, 2*np.pi+np.pi/6, np.pi/6)
        x = center[0] + radius*np.sin(q)
        y = center[1] + radius*np.cos(q)
        plt.plot(x, y, color=color)
