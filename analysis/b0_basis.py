import torch
import numpy as np
from matplotlib import pyplot as plt

'''

Add Gaussians together. Do they form peaks and valleys?

Their means are spaced by a period(?)


'''

'''
mu[n] where mu_n are equally spaced out points on a signal 
animate the gaussian move

'''

x = np.arange(-100,100,step=1)

mu = np.array([-75, 0, 75]).reshape(-1, 1)
sigma = np.array([10,20,10]).reshape(-1, 1)
stats = np.hstack((mu, sigma))
print(stats.shape)

def gaussian(x, stats):
    return np.array([(1/(np.sqrt(2*np.pi)*stat[1])) * np.exp(-0.5 * ((x - stat[0])/stat[1])**2) for stat in stats])


g = gaussian(x, stats)
gsum = np.sum(g, axis=0)
print(g.shape)

integral = np.cumsum(g)
print(f"Integral: {integral[-1]}")
plt.figure()
plt.title("Gaussian Curve")
plt.xlabel("x")
plt.ylabel("probability")
plt.plot(x,g.T)

plt.figure()
plt.title("Gaussian sum")
plt.xlabel("x")
plt.ylabel("probability")
plt.plot(x,gsum.T)

plt.show()



