import numpy as np
import ctypes
import scipy.io as sio
from pymcmcstat.MCMC import MCMC
from numpy.ctypeslib import ndpointer
from pymcmcstat.ParallelMCMC import ParallelMCMC, load_parallel_simulation_results
from pymcmcstat.chain import ChainStatistics as CS
from pymcmcstat.chain import ChainProcessing as CP
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mcmcplot.mcseaborn as mcsns
import glob
import sys
import pandas as pd


#Need to provide directory where chains are saved as command line argument

def main():
	#Name of the directory where I have the parallel chains
	savedir = sys.argv[1]
	chain_names = glob.glob('./' + savedir + '/chain_*.txt',recursive=True)
	print(f'List of chains found are {chain_names}')
	par_chains = [None]*len(chain_names)
	bur_par_chains = [None]*len(chain_names)
	#This is just to store the chain names/number to add an index to the plot
	index = [None]*len(chain_names)
	#Loop over the files and load into a numpy array
	for i in range(len(chain_names)):
		#Gets the chain name from its path
		index[i] = chain_names[i].split('/')[-1].split('.')[0]

		chain = np.loadtxt(chain_names[i],delimiter=" ",unpack=False)
		par_chains[i] = chain
	
		#Burn 50% only need to do it once since all chains are equal length

		if i == 0:
			burnin = int(len(chain)/2)
			#a over here is the actual index with the index value repeating alot of times - this is needed for the plot
			a = np.empty(len(chain)-burnin,dtype = "S10")
			a.fill(index[i])
			#Need this concatenated array for using pymcmcstat parallel chain plotting function
			combined_chain = chain[burnin:,:]

			

		#Need this concatenated array for using pymcmcstat parallel chain plotting function
		if i != 0:
			combined_chain = np.concatenate((combined_chain,chain[burnin:,:]))
			a = np.concatenate((a,np.array((len(chain)-burnin)*[index[i]])))
		bur_par_chains[i] = chain[burnin:,:]

	#Use the pymcmcstat function to get the gelman rubin characteristics
	psrf = CS.gelman_rubin(chains=par_chains,display=True)

	#Gelman Rubin after applying burnin
	print('Gelman Rubin after burnin (50%)\n')
	psrf = CS.gelman_rubin(chains=bur_par_chains,display=True)


	#Plotting the chains on top of each other
	#I am currently unable to make sure each chain has a different color which is very frustrating
	f ,(ax1,ax2) = plt.subplots(2)
	ax1.set_prop_cycle(color=['red','green','blue'])
	ax2.set_prop_cycle(color=['red','green','blue'])
	sns.set_style('whitegrid')

	for ii in range(len(chain_names)):
		df = pd.DataFrame(bur_par_chains[ii],columns=['eta','gamma'])
		sns.kdeplot(data=df[['eta']],ax=ax1)
		sns.kdeplot(data=df[['gamma']],ax=ax2)

	# ax1.legend(index,fontsize=16)
	# ax2.legend(index,fontsize=16)
	plt.savefig('./images/'+savedir+'.png')







	



















if __name__ == "__main__":
    main()