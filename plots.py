#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code used to generate figures
'''

import numpy as np
import pandas as pd
import seaborn as sns
import imageio

from scipy import misc
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numpy.fft import fft2, fftshift

from pspec import *

__author__ = 'Samuel Hulse'
__email__ = 'hsamuel1@umbc.edu'

plt.rcParams["font.family"] = "arial"
plt.rcParams['figure.dpi'] = 600

minor_text_size = 6
major_text_size = 8

blue = '#1f77b4'
orange = '#ff7f0e'

def random_sample(data, sample_dim):
    sample_domain = np.subtract(data.shape[0:2], (sample_dim, sample_dim))       
    if sample_domain[0] == 0: sample_x = 0
    else: sample_x = np.random.randint(0, sample_domain[0])
    if sample_domain[1] == 0: sample_y = 0
    else: sample_y = np.random.randint(0, sample_domain[1])

    sample = data[sample_x: sample_x + sample_dim, sample_y: sample_y + sample_dim]

    return sample

#Habitat Violinplot
f, ax = plt.subplots()

labels = ['Gravel', 'Boulder', 'Bedrock', 'Detritus', 'Sand']

hab_data = pd.read_csv('C:/Users/renoult/Desktop/Sam/habitats.csv')
sns.violinplot(x='habitat', 
    y='slope',
    order=['gravel', 'boulder', 'bedrock', 'detritus', 'sand'],
    inner='quartile',
    linewidth=1,
    color='#abc9ea',
    data=hab_data)

ax.yaxis.grid(linestyle=':')
ax.set_xticklabels(labels, rotation=-45)
ax.tick_params(axis='both', labelsize=minor_text_size)
ax.set_xlabel('Habitat Class', fontsize=major_text_size)
ax.set_ylabel('Fourier Slope', fontsize=major_text_size)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('../Figures/Figure 4.svg')

#Darter Violinplot
f, ax = plt.subplots()

colors = ['#1f77b4', '#ff7f0e']
labels = ['E. caeruleum','E. zonale','E. blennioides','E. camurum','E. barrenense',
        'E. swaini','E. gracile','E. olmstedi','E. pyrrhogaster','E. chlorosomum']

fish_data = pd.read_csv('C:/Users/renoult/Desktop/Sam/fish.csv')
darters = sns.violinplot(x='species',
    y='slope',
    hue='sex',
    order=['caeruleum','zonale','blennioides','camurum','barrenense',
        'swaini','gracile','olmstedi','pyrrhogaster','chlorosomum'],
    split=True,
    inner='quartile',
    linewidth=1,
    palette='pastel',
    data=fish_data)

rect1 = patches.Rectangle((-0.5,-5), 2, 10, alpha=0.1, facecolor='k', zorder=0)
rect2 = patches.Rectangle((2.5,-5), 2, 10, alpha=0.1, facecolor='k', zorder=0)
rect3 = patches.Rectangle((6.5,-5), 3, 10, alpha=0.1, facecolor='k', zorder=0)
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

ax.text(0.5, -1.15, 'gravel', rotation=-45, ha='center', fontsize=minor_text_size)
ax.text(2, -1.15, 'bedrock', rotation=-45, ha='center', fontsize=minor_text_size)
ax.text(3.5, -1.15, 'boulder', rotation=-45, ha='center', fontsize=minor_text_size)
ax.text(5.5, -1.15, 'detritus', rotation=-45, ha='center', fontsize=minor_text_size)
ax.text(8, -1.15, 'sand', rotation=-45, ha='center', fontsize=minor_text_size)

ax.yaxis.grid(linestyle=':')
ax.set_xticklabels(labels, rotation=-45, style='italic')
ax.tick_params(axis='both', labelsize=minor_text_size)

ax.set_ylim(-4.55, -1)

ax.set_xlabel('Species', fontsize=major_text_size)
ax.set_ylabel('Fourier Slope', fontsize=major_text_size)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.legend(title=None, loc=8, ncol=2, frameon=False)
plt.tight_layout()
plt.savefig('../Figures/Figure 3.svg')

#Scatterplots
fish_data = pd.read_csv('C:/Users/renoult/Desktop/Sam/fish.csv')
hab_data = pd.read_csv('C:/Users/renoult/Desktop/Sam/habitats.csv')

habitats = {}
habitats['barrenense'] = 'bedrock'
habitats['blennioides'] = 'boulder'
habitats['caeruleum'] = 'gravel'
habitats['camurum'] = 'boulder'
habitats['chlorosomum'] = 'sand'
habitats['gracile'] = 'detritus'
habitats['olmstedi'] = 'sand'
habitats['pyrrhogaster'] = 'sand'
habitats['swaini'] = 'detritus'
habitats['zonale'] = 'gravel'

means = fish_data.groupby(['species', 'sex']).mean()
m_vals = means.query('sex == "M"').values[:,0]
f_vals = means.query('sex == "F"').values[:,0]
h_vals = means.query('sex == "M"').values[:,1]

sem_fish = fish_data.groupby(['species', 'sex']).sem()
sem_m = sem_fish.query('sex == "M"').values[:,4]
sem_f = sem_fish.query('sex == "F"').values[:,4]
sem_hab_vals = hab_data.groupby(['habitat']).sem().values

s_list = fish_data['species'].drop_duplicates().values
habs = [habitats[x] for x in s_list]

h_list = hab_data['habitat'].drop_duplicates().values

h_dict = dict(zip(h_list, sem_hab_vals))
sem_h = np.array([h_dict[x] for x in habs])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
plt.subplots_adjust(wspace=0.3)

ax[0].errorbar(h_vals,
    m_vals,
    xerr=sem_h,
    yerr=sem_m,
    ms=4,
    alpha=0.7,
    capsize=2,
    capthick=1,
    color=blue,
    fmt='o')
ax[1].errorbar(h_vals,
    f_vals,
    xerr=sem_h,
    yerr=sem_f,
    ms=4,
    alpha=0.7,
    capsize=2,
    capthick=1,
    color=orange,
    fmt='o')

z = np.polyfit(m_vals, h_vals, 1)
p = np.poly1d(z)

ss_res = np.sum((p(m_vals) - np.mean(h_vals))**2)
ss_tot = np.sum((h_vals - np.mean(h_vals))**2)
r2_m = 1 - ss_res/ss_tot

w = np.polyfit(f_vals, h_vals, 1)
j = np.poly1d(w)

ss_res = np.sum((p(f_vals) - np.mean(h_vals))**2)
ss_tot = np.sum((h_vals - np.mean(h_vals))**2)
r2_f = 1 - ss_res/ss_tot

x=[-5, -2]

ax[0].plot(x, p(x), color=blue)
ax[1].plot(x, j(x), color=orange)

ax[0].set_xlim(-3.55, -2.75)
ax[0].set_ylim(-3.5, -2.5)
ax[1].set_xlim(-3.55, -2.75)
ax[1].set_ylim(-3.5, -2.5)


ax[0].tick_params(axis='both', labelsize=minor_text_size)
ax[1].tick_params(axis='both', labelsize=minor_text_size)

ax[0].set_ylabel('Darter Slope', fontsize=major_text_size)
ax[0].set_xlabel('Habitat Slope', fontsize=major_text_size)
ax[1].set_ylabel('Darter Slope', fontsize=major_text_size)
ax[1].set_xlabel('Habitat Slope', fontsize=major_text_size)

ax[0].annotate('a', (-0.2, 1), xycoords='axes fraction', fontsize=14)
ax[1].annotate('b', (-0.2, 1), xycoords='axes fraction', fontsize=14)

ax[0].annotate('*', (h_vals[1]-0.013, m_vals[1]+0.08), fontsize=14)
ax[0].annotate('*', (h_vals[7]-0.013, m_vals[7]+0.08), fontsize=14)


'''
ax[0].annotate('$r^2$: ' + "%.3f" % round(r2_m, 3), (0.1, 0.90), xycoords='axes fraction', fontsize=10)
ax[1].annotate('$r^2$: ' + "%.3f" % round(r2_f, 3), (0.1, 0.90), xycoords='axes fraction', fontsize=10)
'''

ax[0].grid(linestyle=':')
ax[1].grid(linestyle=':')

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
plt.savefig('../Figures/Figure 5.svg')

#Slope Demo Plot
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

sand = imageio.imread('C:/Users/renoult/Desktop/Sam/Code/sand.jpg')
boulder = imageio.imread('C:/Users/renoult/Desktop/Sam/Code/boulder.jpg')

ax[0,0].imshow(sand)
ax[1,0].imshow(boulder)
ax[0,0].get_yaxis().set_ticks([])
ax[0,0].get_xaxis().set_ticks([])
ax[1,0].get_yaxis().set_ticks([])
ax[1,0].get_xaxis().set_ticks([])
ax[0,0].annotate('sand', (-0.1, 0.5), xycoords='axes fraction', fontsize=12, rotation=90, va='center')
ax[1,0].annotate('boulder', (-0.1, 0.5), xycoords='axes fraction', fontsize=12, rotation=90, va='center')
ax[0,0].spines["top"].set_visible(False)
ax[0,0].spines["right"].set_visible(False)
ax[0,0].spines["bottom"].set_visible(False)
ax[0,0].spines["left"].set_visible(False)
ax[1,0].spines["top"].set_visible(False)
ax[1,0].spines["right"].set_visible(False)
ax[1,0].spines["bottom"].set_visible(False)
ax[1,0].spines["left"].set_visible(False)

sand = sand[:,:,0]*0.21 + sand[:,:,1]*0.72 + sand[:,:,2]*0.07
boulder = boulder[:,:,0]*0.21 + boulder[:,:,1]*0.72 + boulder[:,:,2]*0.07
sand = misc.imresize(sand, (800, 1200))
boulder = misc.imresize(boulder, (800, 1200))
sand = random_sample(sand, 400)
boulder = random_sample(boulder, 400)
sand = kaiser2D(sand, 2)
boulder = kaiser2D(boulder, 2)

s_fft = imfft(sand)
s_x = np.linspace(1, len(s_fft), len(s_fft))

b_fft = imfft(boulder)
b_x = np.linspace(1, len(b_fft), len(b_fft))

ax[0,1].loglog(s_x, s_fft, color='k', lw=0.5)
ax[1,1].loglog(b_x, b_fft, color='k', lw=0.5)

s_bin_x, s_bin_y = bin_pspec(s_fft, 20, (10,200))
b_bin_x, b_bin_y = bin_pspec(b_fft, 20, (10,200))
ax[0,1].scatter(s_bin_x, s_bin_y, alpha=0.75, color=orange, s=10, marker='x', zorder=3)
ax[1,1].scatter(b_bin_x, b_bin_y, alpha=0.75, color=orange, s=10, marker='x', zorder=3)

s_slope = get_slope(s_bin_x, s_bin_y)
b_slope = get_slope(b_bin_x, b_bin_y)

ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False)
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False)

def plot_exp(vals, x, y):
    x, y = np.log((x, y))
    slope, int = np.polyfit(x, y, 1)
    return (np.exp(int)*(vals**slope))

ax[0,1].plot(s_bin_x, plot_exp(s_bin_x, s_bin_x, s_bin_y), color=blue, linewidth=2, ls='--')
ax[1,1].plot(b_bin_x, plot_exp(b_bin_x, b_bin_x, b_bin_y), color=blue, linewidth=2, ls='--')

ax[0,1].grid(linestyle=':')
ax[1,1].grid(linestyle=':')

ax[0,1].set_ylim(10**-6, 10**4.5)
ax[1,1].set_ylim(10**-6, 10**4.5)


ax[0,1].annotate('slope: ' + "%.3f" % round(s_slope, 3), (0.65, 0.8), xycoords='axes fraction', fontsize=major_text_size)
ax[1,1].annotate('slope: ' + "%.3f" % round(b_slope, 3), (0.65, 0.8), xycoords='axes fraction', fontsize=major_text_size)

ax[0,1].tick_params(axis = 'both', which = 'major', labelsize = minor_text_size)
ax[1,1].tick_params(axis = 'both', which = 'major', labelsize = minor_text_size)
ax[0,1].set_xlabel('Frequency (cycles per image)', fontsize=major_text_size)
ax[1,1].set_xlabel('Frequency (cycles per image)', fontsize=major_text_size)
ax[0,1].set_ylabel('Power', fontsize=major_text_size)
ax[1,1].set_ylabel('Power', fontsize=major_text_size)

plt.savefig('../Figures/Figure 2.svg')
