import numpy as np
path = "/home/sean/PrisonerEscape/simulator/forest_coverage/map_set/2.npy"

m = np.load(path)

m[m < 0.45] = 0.2

np.save('simulator/forest_coverage/map_set/2_darker.npy', m)