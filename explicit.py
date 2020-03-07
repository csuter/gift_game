import numpy as np
from pprint import pprint
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=1000)


def draw_state(ax, x, y, l, r, p, n, state_spread=.05):
  angles = 2 * np.pi * np.arange(n, dtype=np.float32) / n
  xys = state_spread * np.stack([np.sin(angles), -np.cos(angles)], axis=-1)
  for i in range(n):
    e = mpl.patches.Ellipse(
        xy=xys[i, :] + [x, y],
        height=state_spread / 3,
        width=state_spread / 3,
        facecolor='w' if i != p else 'k',
        edgecolor='k',
        linewidth=.5)
    ax.add_artist(e)

  # right tri
  # hyp = rad
  # angle = 180 / n
  # side len = 2 * sin(angle)
  side_len = 2 * state_spread * np.sin(np.pi / n)
  enclosure_radius = .9 * side_len / 2

  lcirc = mpl.patches.Arc(
      xy = xys[l, :] + [x, y],
      height=enclosure_radius * 2,
      width=enclosure_radius * 2,
      edgecolor='k',
      facecolor='#00000000',
      linewidth=.5,
      theta1=180 * angles[l] / np.pi + 90,
      theta2=180 * angles[l] / np.pi - 90)

  rcirc = mpl.patches.Arc(
      xy = xys[r, :] + [x, y],
      height=enclosure_radius * 2,
      width=enclosure_radius * 2,
      edgecolor='k',
      facecolor='#00000000',
      linewidth=.5,
      theta1=180 * angles[r] / np.pi - 90,
      theta2=180 * angles[r] / np.pi + 90)

  incirc = mpl.patches.Arc(
      xy = [x, y],
      height=(state_spread - enclosure_radius) * 2,
      width=(state_spread - enclosure_radius) * 2,
      edgecolor='k',
      facecolor='#00000000',
      linewidth=.5,
      theta1=180 * angles[l] / np.pi - 90,
      theta2=180 * angles[r] / np.pi - 90)

  outcirc = mpl.patches.Arc(
      xy = [x, y],
      height=(state_spread + enclosure_radius) * 2,
      width=(state_spread + enclosure_radius) * 2,
      edgecolor='k',
      facecolor='#00000000',
      linewidth=.5,
      theta1=180 * angles[l] / np.pi - 90,
      theta2=180 * angles[r] / np.pi - 90)

  ax.add_artist(lcirc)
  ax.add_artist(rcirc)
  ax.add_artist(incirc)
  ax.add_artist(outcirc)


def left_n(p, n):
  return (p - 1) % n

def right_n(p, n):
  return (p + 1) % n

def get_neighbors_n(l, r, p, n):
  left = lambda p: left_n(p, n)
  right = lambda p: right_n(p, n)
  if p == l and p == r:
    return (left(l), r, left(p)), (l, right(r), right(p))
  if p == l:
    return (left(l), r, left(p)), (l, r, right(p))
  if p == r:
    return (l, r, left(p)), (l, right(r), right(p))
  return (l, r, left(p)), (l, r, right(p))

for n in range(4, 5):
  print("n:", n)
  left = lambda p: left_n(p, n)
  right = lambda p: right_n(p, n)

  states_by_id = []
  ids_by_state = {}
  states_by_group_size = {}

  for size in range(1, n):
    states_by_group_size[size] = []
    for j in range(-size + 1, 1):
      l = j % n
      r = (l + size - 1) % n
      for k in range(j, r + 1):
      #for k in (range(j, r + 1) if size % 2 == 0 else range(r, j - 1, -1)):
        p = k % n
        entry = l, r, p
        states_by_group_size[size].append(entry)
        ids_by_state[entry] = len(states_by_id)
        states_by_id.append(entry)

  states_by_group_size[n] = []
  for entry in states_by_group_size[n - 1]:
    l, r, p = entry
    if p == l:
      entry = left(l), r, left(p)
      states_by_group_size[n].append(entry)
      ids_by_state[entry] = len(states_by_id)
      states_by_id.append(entry)
    if p == r:
      entry = l, right(r), right(p)
      states_by_group_size[n].append(entry)
      ids_by_state[entry] = len(states_by_id)
      states_by_id.append(entry)

#  max_num = np.max([len(states_by_group_size[size]) for size in states_by_group_size])
#
#  fig = plt.figure(figsize=(15, 15))
#  ax = fig.add_axes((0, 0, 1, 1))
#
#  for size in range(1, n + 1):
#    num_entries = len(states_by_group_size[size])
#    for j, entry in enumerate(states_by_group_size[size]):
#      x = (j + 1) * 1. / (num_entries + 1)
#      y = size * 1. / (n + 1)
#      l, r, p = entry
#      draw_state(ax, x, y, l, r, p, n=n, state_spread=.2 / (max_num + 1))
#  plt.savefig('viz_{}.png'.format(n))


  #print("states_by_group_size", states_by_group_size)
  #print("states_by_id", states_by_id)
  #print("ids_by_state", ids_by_state)
  transition_matrix = np.zeros([len(states_by_id), len(states_by_id)])
  get_neighbors = lambda *state: get_neighbors_n(*state, n)
  for i, state in enumerate(states_by_id):
    #print(i, state)
    l, r, p = state
    if right(r) == l and left(l) == r:
      transition_matrix[i, i] = 1
      continue
    n1, n2 = get_neighbors(*state)
    n1_id = ids_by_state[n1]
    n2_id = ids_by_state[n2]
    transition_matrix[n1_id, i] = .5
    transition_matrix[n2_id, i] = .5
  #print(transition_matrix)

  #fig = plt.figure(figsize=(10, 10))
  #plt.imshow(transition_matrix)
  #plt.yticks(range(0, len(states_by_id)))
  #plt.xticks(range(0, len(states_by_id)))
  ##plt.xticks(range(0, len(states_by_id)))
  #plt.savefig('txmat_{}.png'.format(n))
  #plt.close(fig)

  q = transition_matrix[:-2 * (n - 1), :-2 * (n - 1)]
  iq = np.eye(q.shape[0])

  print(q)

  r = transition_matrix[-2 * (n - 1):, :-2 * (n - 1)]
  print(r)

  print(r.shape)
  print((iq - q).shape)

  iq_m_q_inv = np.linalg.inv(iq - q)

  print(iq_m_q_inv)
  print(iq_m_q_inv[:, 0])

  win_dist = r.dot(iq_m_q_inv[:, 0])

  #win_dist = t_inf_bl[:, 0]
  print(win_dist)

  # |1 1 0 0 0 0    ...|
  # |0 0 1 1 0 0       |
  # |0 0 0 0 1 1       |
  # |...        ...    |
  # |...        1 1 0 0|
  # |...            1 1|
  proj = np.kron(np.eye(n), np.ones([2]))
  # |1 0 0 0 0    ...|
  # |0 1 1 0 0       |
  # |0 0 0 1 1       |
  # |...      ...    |
  # |...        1 1 0|
  # |...          ..1|
  proj = proj[:, 1:-1]
  print(proj.dot(win_dist))

  #fig = plt.figure(figsize=(10, 10))
  #plt.imshow(q)
  #plt.savefig('q_{}.png'.format(n))
  #plt.close(fig)
