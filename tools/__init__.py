import os
import re
from copy import copy

home_dir = os.environ["HOME"]
current_user = os.environ["USER"]


DA = 10 						#arbor diameter (default: 19 pixel)
N = 16

if current_user=="bh2757":
	DA = 19
	N = 64
x_LGN,y_LGN = N,N
x_cortex,y_cortex = N,N
	
		
params = dict(
		DA = DA,					#arbor diameter
		rA = (DA - 0)/2.,			#arbor radius
		cA = 0.5,
		gammaC = 3,					#rel width I to E (corr)
		gammaI = 3,					#rel width I to E (rec interaction)
		aI = 0.5,					#always < 1
		target_sigma = 0.01,		#for estimation of growth factor lambda\
		#determines SD of change in S in one iteration
		PBC = True,					#periodic boundary conditions
		s_max = 12,					#maximum strength of synapse
		s_noise = 0.2,				#for initializing ff weights S
		lambda_init = 0.01,			#initial growth factor
		N = x_cortex				# system size
)

LGN_corr_params = {
					"ampl"		:	1.,
					"profile"	:	"Gaussian",
}
interaction_params = {
					"ampl"		:	1.,
					"profile"	:	"Gaussian",
}

if params['PBC']:
	assert x_LGN==x_cortex, "x_LGN!=x_cortex"
	assert y_LGN==y_cortex, "y_LGN!=y_cortex"


##paths
if current_user=="bettina":
	home_dir = home_dir + "/physics/columbia/projects/miller94/update_clean/"
elif current_user=="bh2757":
	home_dir = "/burg/theory/users/bh2757/columbia/projects/miller94/update_clean/"
data_dir = home_dir+'data/'
image_dir = home_dir+'image/'
movie_dir = home_dir+'movie/'