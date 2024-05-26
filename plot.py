#!/usr/bin/env python3

from matplotlib.pyplot import *
from math import *
import numpy as np

name=sys.argv[1]

with open(name, "r") as f:
    contenu = f.read().split("\n\n")
    # print(contenu)

x = contenu[0].split()
nx = len(x) - 1
x = np.array([float(x[i]) for i in range(nx+1)])
# print(x)
y = contenu[1].split()
ny = len(y) -1
y = np.array([float(y[i]) for i in range(ny+1)])
# print(y)
x , y = np.meshgrid(x,y)
z = contenu[2].split()
nz = len(z)
z = np.array([float(z[i]) for i in range(nz)]).reshape((ny+1,nx+1))
# print(z)
# plot(xi, unum, color="blue")
# plot(xi, uex, color="red")
# def quit_figure(event):
#     if event.key == 'q':
#         close(event.canvas.figure)
# cid = gcf().canvas.mpl_connect('key_press_event', quit_figure)

min_val = np.min(z)
max_val = np.max(z)

print(z.shape)

print('min = '+str(min_val))
print('max = '+str(max_val))

fig, ax = subplots(figsize=(12, 6))
#cs = ax.contourf(x,y,z,100)
cs = ax.pcolormesh(x,y,z,cmap='jet')
cbar = fig.colorbar(cs, ax=ax)
ax.set_aspect('equal', 'box')


print("press \'q\' to quit...");
show()
print("The end")
