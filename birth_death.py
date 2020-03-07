import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

num_parallel = 100000
num_flips = 200
flips = np.random.choice([-1, 1], size=[num_parallel, num_flips])

cumsum = np.concatenate([np.zeros([num_parallel, 1]), np.cumsum(flips, axis=-1)], axis=-1)

lims = np.arange(1, 11)
n_lims = lims.shape[0]

bigger = cumsum >= lims[..., None, None]
first = np.argmax(bigger, axis=-1)
first[first==0] = 2**32 - 1
print(first.shape)

fig = plt.figure(figsize=(20, 4 * n_lims))
axes = fig.subplots(nrows=n_lims)

for i, lim in enumerate(lims):
  axes[i].hist(first[i, :], density=True, label='limit = {}'.format(lim), bins=np.arange(200))
  axes[i].legend()
  axes[i].set_xlim(0, 200)
plt.savefig('firsts.png')
