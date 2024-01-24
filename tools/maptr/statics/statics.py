import matplotlib.pyplot as plt
import numpy as np
import pdb


d_curvature_target = []
with open('delta_derivatives_degree_target2.txt', 'r') as f:
    for line in f.readlines():
        l = line.strip().split(' ')
        l = np.array([float(i) for i in l])
        d_curvature_target.extend(l)
    
d_curvature_target = np.array(d_curvature_target)

d_curvature_target_ = d_curvature_target[abs(d_curvature_target)<0.3]
print(d_curvature_target_.shape)
print(d_curvature_target.shape)
print(d_curvature_target_.shape[0]/d_curvature_target.shape[0])

# Creating a histogram
plt.figure(figsize=(10, 6))
plt.hist(d_curvature_target, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Changes in Values of Slope (in radians)',fontsize=20)
plt.xlabel('Value',fontsize=20)
plt.ylabel('Frequency',fontsize=20)
plt.xlim([-np.pi/2,np.pi/2])
plt.xticks(np.linspace(-np.pi/2,np.pi/2,10).tolist())
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

plt.grid(True)
plt.show()
plt.savefig('delta_derivatives_degree_target2.jpg')

