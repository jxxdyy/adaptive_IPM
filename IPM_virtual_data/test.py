import numpy as np
import math as m
from scipy.spatial.transform import Rotation as R

z = np.deg2rad(20)
y = np.deg2rad(10)
x = np.deg2rad(30)

Rz = np.array([[np.cos(z), -np.sin(z), 0],
               [np.sin(z), np.cos(z), -0],
               [0, 0, 1]])

Ry = np.array([[np.cos(y), 0 ,np.sin(y)],
               [0, 1, 0],
               [-np.sin(y), 0, np.cos(y)]])

Rx = np.array([[1, 0, 0],
               [0, np.cos(x), -np.sin(x)],
               [0, np.sin(x), np.cos(x)]])


print(Ry)

# print(np.dot(Rz, np.dot(Ry, Rx)))
# print(Rz@(Ry@Rx))
# print('------------')
# print(np.dot(Rx, np.dot(Ry, Rz)))
# print(Rx@(Ry@Rz))
print('------------')
# print(np.dot(Rx, Rz))
# print(np.matmul(Rx, Rz))
# print('------------')

rz = R.from_rotvec([0, 0, z])
Rz2 = rz.as_matrix()


rx = R.from_rotvec([x, 0, 0])
Rx2 = rx.as_matrix()


rxz = R.from_rotvec([0, y, 0])
Rxz2 = rxz.as_matrix()
print(Rxz2)