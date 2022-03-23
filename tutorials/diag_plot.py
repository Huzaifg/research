import matplotlib.pyplot as mpl
import theano.tensor as tt
import pymc3 as pm
import arviz as az
import os
import sys
#Import our 8dof vehicle model
from vd_bi_mod import vehicle_bi


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
	var_names = ['Cf','Cr','a','b',"sigmaLat_acc","sigmaVy"]

	#Read the data file
	idata = az.from_netcdf('./results/' + filename + ".nc")
	# idata = az.from_netcdf('./euler_outs/' + filename + ".nc")
	#Plot the posterior





	ax_post = az.plot_posterior(idata,var_names = var_names)
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
	ax_trace = az.plot_trace(idata,var_names = var_names,figsize = (14,6*2))
	fig = ax_trace.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_trace.png",facecolor = 'w')
	mpl.show()

	# Plot pairwise plot
	ax_pair = az.plot_pair(idata,var_names = var_names,figsize = (14,6*2))
	fig = ax_pair.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_pair.png",facecolor = 'w')
	mpl.show()


	#Autocorrelation plot
	ax_autocorr = az.plot_autocorr(idata,var_names = var_names)
	fig = ax_autocorr.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_autocorr.png",facecolor = 'w')
	mpl.show()


	ax_ess = az.plot_ess(idata,var_names = var_names,kind = "evolution")
	fig = ax_ess.ravel()[0].figure
	if(save):
		fig.savefig(path + filename + "_ess.png",facecolor = 'w')
	mpl.show()

















if __name__ == "__main__":
	main()

