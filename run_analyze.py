import numpy as np
import matplotlib.pyplot as plt


log = np.load('log.npy')

map_mean = []
map_50 = []
map_75 = []

for i in range(len(log)):
    map_mean.append(log[i][0])
    map_50.append(log[i][1])
    map_75.append(log[i][2])

plt.plot(map_mean, color='r', label='map_mean')
plt.plot(map_50, color='b', label='map_50')
plt.plot(map_75, color='g', label='map_75')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode='expand', borderaxespad=0.)

map_mean = np.around(map_mean, decimals=4)
map_50 = np.around(map_50, decimals=4)
map_75 = np.around(map_75, decimals=4)

print('map_mean')
print(map_mean)
print('map_50')
print(map_50)
print('map_75')
print(map_75)
plt.show()

