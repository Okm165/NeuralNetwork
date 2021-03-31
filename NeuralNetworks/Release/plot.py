import matplotlib.pyplot as plt
import numpy as np
T = []
file = open("output.txt", "r")
lines = file.read().splitlines()

for line in lines:
    T.append(float(line))

ypoints = T
plt.plot(ypoints)
plt.show()
