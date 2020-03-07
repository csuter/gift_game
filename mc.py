import scipy.stats
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import patches
import matplotlib.pyplot as plt
np.random.seed(42)

num_players = 6  # Including host
num_parallel = 50000
num_flips = 3 * (num_players ** 2)

zeros = np.zeros([num_parallel, 1], np.int32)
def pad(x):
  return np.concatenate([zeros, x], axis=-1)

flips = np.random.choice([-1, 1], size=[num_parallel, num_flips])
positions = pad(np.cumsum(flips, axis=-1))

cum_min = np.minimum.accumulate(positions, axis=-1)
cum_max = np.maximum.accumulate(positions, axis=-1)
cum_max_min_diff = (cum_max - cum_min)

gameover = np.int32(cum_max_min_diff >= num_players - 2)
gameended = pad(gameover[..., 1:] - gameover[..., :-1])

game_end_steps = np.argmax(gameended, axis=-1)

maybe_winners = pad((positions[..., 1:] + flips) % num_players)
winners = np.array([maybe_winners[i, s] for i, s in enumerate(game_end_steps)])

print(winners.shape)
print(game_end_steps.shape)

mean_steps = game_end_steps.mean()
min_steps = game_end_steps.min()
max_steps = game_end_steps.max()
end_bins = np.arange(min_steps, max_steps + 1)


fig = plt.figure(figsize=(20, 3 * num_players))
axes = fig.subplots(nrows=num_players-1)
for i in range(1, num_players):
  axes[i - 1].hist(
      game_end_steps[winners == i],
      alpha=.5,
      label='winner = {}'.format(i),
      bins=end_bins)
  axes[i - 1].set_xlim(0, num_players ** 2)
  axes[i - 1].legend()
plt.savefig('{}_player_per_winner_game_lengths.png'.format(num_players))

fig = plt.figure(figsize=(12, 6))
axes = fig.subplots(ncols=2)

axes[0].hist(winners, bins=np.arange(1, num_players+1), edgecolor='k')
axes[0].set_ylabel("Number of wins")
axes[0].set_xlabel("Winning position")
axes[0].set_title("Histogram of winning positions across {} repetitions".format(num_parallel))
axes[0].set_xticks(np.arange(1, num_players) + .5)
axes[0].set_xticklabels(np.arange(1, num_players))

axes[1].hist(game_end_steps, label='Game length', bins=end_bins, density=True)
axes[1].axvline(mean_steps, c='r', label='Mean = {}'.format(mean_steps))
axes[1].axvline(min_steps, c='b', label='Min = {}'.format(min_steps))
axes[1].axvline(max_steps, c='g', label='Max = {}'.format(max_steps))
#axes[1].plot(end_bins, scipy.stats.poisson.pmf(end_bins, mean_steps, 0), c='k')

axes[1].legend()
plt.savefig('{}_player_winner_dist_and_game_lengths.png'.format(num_players))
