import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from batchfd import get_fd

fish_dir = '/media/sam/Seagate Expansion Drive/Images/Crops'
hab_dir = '/media/sam/Seagate Expansion Drive/Images/Catagorized'

'''
data_m = get_fd(fish_dir, selection = 'M')
data_m = data_m.drop(['punctulatum'], axis=1)
data_m = data_m.reindex_axis(data_m.mean().sort_values().index, axis=1)
data_f = get_fd(fish_dir, selection = 'F')
data_f = data_f.drop(['punctulatum'], axis=1)
data_f = data_f.reindex_axis(data_f.mean().sort_values().index, axis=1)
'''
data_h = get_fd(hab_dir, resize_dim = [900, 600], sample_dim = 200)
