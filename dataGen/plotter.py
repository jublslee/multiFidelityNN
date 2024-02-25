# plotter scripts
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import norm
import numpy as np
import os

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

# plotting specs
fs = 24
plt.rc('font',  family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text',  usetex=True)



#--------------------------------------------------------------------------#
# contour plotting for the 2D reaction-diffusion system
# Inputs:
#        path: where to save the picture
#        name: name of the figure
#        data: solution tensor
#        t_idx: which time instance to plot 
#        colorbar: if true, add color bar
#        cbar_range: if colorbar == True, specify range of the color bar
#        ticksoff: if False, remove ticks
def RD_plotter(path, name, data, t_idx, colorbar=False, cbar_range = None, ticksoff=False):
	
	os.makedirs(path,exist_ok = True)
	
	# c1
	fig, ax = plt.subplots(figsize=(10, 10))
	im = plt.imshow(
		data[t_idx, ..., 0].transpose(),
		aspect="auto",
		origin="lower",
		extent=[
		-1,
		1,
		-1,
		1,
		],
		)
	plt.tick_params(labelsize=fs-4)

	# if not include ticks
	if ticksoff == True:
		plt.xticks([])
		plt.yticks([])

	if ticksoff == False:
		plt.xlabel("$x$",fontsize=fs)
		plt.ylabel("$y$",fontsize=fs)

	if colorbar == True:
		im.set_clim(cbar_range[0][0],cbar_range[0][1])
		cbar = plt.colorbar(im)
		cbar.ax.tick_params(labelsize=fs+12)
		cbar.ax.locator_params(nbins=5)
	plt.savefig(path + name + '-c1.pdf', bbox_inches='tight',pad_inches = 0)
	
	# c2
	fig, ax = plt.subplots(figsize=(10, 10))
	im = plt.imshow(
		data[t_idx, ..., 1].transpose(),
		aspect="auto",
		origin="lower",
		extent=[
		-1,
		1,
		-1,
		1,
		],
		)
	plt.tick_params(labelsize=fs-4)

	# if not include ticks
	if ticksoff == True:
		plt.xticks([])
		plt.yticks([])

	if ticksoff == False:
		plt.xlabel("$x$",fontsize=fs)
		plt.ylabel("$y$",fontsize=fs)

	if colorbar == True:
		im.set_clim(cbar_range[1][0],cbar_range[1][1])
		cbar = plt.colorbar(im)
		cbar.ax.tick_params(labelsize=fs+12)
		cbar.ax.locator_params(nbins=5)
	plt.savefig(path + name + '-c2.pdf', bbox_inches='tight',pad_inches = 0)
	
	return 0
#--------------------------------------------------------------------------#
