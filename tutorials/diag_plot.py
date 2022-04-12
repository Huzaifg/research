import matplotlib.pyplot as mpl
import pymc as pm
import arviz as az
import os
import sys
import numpy as np
from bokeh.io import export_png
from bokeh.layouts import gridplot



"""
Command line inputs required
filename : The filename of the inference data file generated after mcmc. DO NOT prepend the results subdirectory or the file extension
save : True/False - Whether the plots need to be saved to ./images


returns 
Returns : 
Opens the plots on the screen
If save is true, then the plots will also be saved in the ./images subdirectory 

"""

def main():
	filename = sys.argv[1]
	save = sys.argv[2].lower() == 'true'
	# Need to always keep changing these varnames which is a pain
	var_names = ['Cf','Cr','m',"sigmaLat_acc","sigmaVy"]

	#Read the data file
	idata = az.from_netcdf('./results/' + filename + ".nc")
	# idata = az.from_netcdf('./euler_outs/' + filename + ".nc")
	#Plot the posterior





	ax_post = az.plot_posterior(idata)
	fig = ax_post.ravel()[0].figure

	path = 'images/'
	if(os.path.isdir(path)):
		if(save):
			fig.savefig(path + filename + "_post.png",facecolor = 'w')
	else:
		os.mkdir(path)
		if(save):
			fig.savefig(path + filename + "_post.png",facecolor = 'w')
	
	mpl.show()

	#Plot the trace
	ax_trace = az.plot_trace(idata,backend = "bokeh",figsize = (12,14))
	if(save):
		export_png(gridplot(ax_trace.tolist()), filename=path + filename + "_trace.png")

	# fig = ax_trace.ravel()[0].figure
	# if(save):
	# 	fig.savefig(path + filename + "_trace.png",facecolor = 'w')
	# mpl.show()

	# Plot pairwise plot
	ax_pair = az.plot_pair(idata,divergences = True,figsize = (14,6*2))
	fig = ax_pair.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_pair.png",facecolor = 'w')
	mpl.show()


	#Autocorrelation plot
	ax_autocorr = az.plot_autocorr(idata,combined = True,figsize = (14,12),textsize = 14)
	fig = ax_autocorr.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_autocorr.png",facecolor = 'w')
	mpl.show()


	ax_ess = az.plot_ess(idata,kind = "evolution")
	fig = ax_ess.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_ess.png",facecolor = 'w')
	mpl.show()


	try:
		print(f"Sampling time in hours {round(idata.sample_stats.sampling_time/3600,2)}")
		print(f"The number of draws are {idata.posterior.draw.shape[0]}")
		print(f"The Bulk relative effective sample size is {az.ess(idata,relative = True)}")
		print(f"The Bulk Effective samples per second is {az.ess(idata)/idata.sample_stats.sampling_time}")	
		print(f"Mean Acceptance rate {np.mean(idata['sample_stats']['acceptance_rate'])}")
		print(f"Mean Tree Depth {np.mean(idata['sample_stats']['tree_depth'])}")
		print(f"Mean Step size {np.mean(idata['sample_stats']['step_size'])}")
		divergences = np.sum([idata['sample_stats']['diverging'][:] == True])
		percent = (divergences)/ idata['sample_stats']['diverging'].shape[1]
		print(f"The number of divergences is {divergences}")
		print(f"The percentage of divergences is {round(percent,5)}")
	except:
		print(f"Sampling time in hours {round(idata.sample_stats._t_sampling/3600,2)}")
		print(f"The number of draws are {idata.posterior.draw.shape[0]}")
		print(f"The Bulk relative effective sample size is {az.ess(idata,relative = True)}")
		print(f"The Bulk Effective samples per second is {az.ess(idata)/idata.sample_stats._t_sampling}")	
		print("You probably ran Metropolis and not NUTS")
		return

















if __name__ == "__main__":
	main()

