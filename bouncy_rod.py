import numpy as np
np.set_printoptions(linewidth=1000)

def bouncy_rod(n):
  tmat = np.zeros([n, n])
  tmat[np.arange(1,n), np.arange(0,n-1)] = .5
  tmat[np.arange(0,n-1), np.arange(1,n)] = .5
  return tmat


for d in range(3, 20):
  eye = np.eye(d)
  tmat = bouncy_rod(d)
  c = (d + 1) / 2.
  print(c * np.linalg.inv(eye - tmat))
  print()



q3 = np.array(
    [[0., 0., 0., 0., 0.],
     [.5, 0., .5, 0., 0.],
     [0., .5, 0., 0., 0.],
     [0., 0., 0., 0., .5],
     [.5, 0., 0., .5, 0.]])

print(np.eye(5) - q3)
print(((5 + 1) / 2) * np.linalg.inv(np.eye(5) - q3))
