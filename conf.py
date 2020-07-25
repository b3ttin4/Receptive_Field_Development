import os
import bettina.modeling.tools.parse_args as pargs

DA = 19

params = dict(
	DA = DA,					#arbor diameter
	rA = (DA -1)/2.,			#arbor radius
	cA = 0.5,
	gammaC = 3,					#rel width I to E (corr)
	gammaI = 3,					#rel width I to E (rec interaction)
	aI = 0.5,					#always < 1
	target_sigma = 0.01,		#for estimation of growth factor lambda\
	#determines SD of change in S in one iteration
	PBC = True,					#periodic boundary conditions
	s_max = 4,					#maximum strength of synapse
	s_noise = 0.2,				#for initializing ff weights S
	lambda_init = 0.01			#initial growth factor
	#sigma_delta = 0.01			
)

debug = pargs.ask_for_mod()
scale = pargs.ask_for_scale()
if debug:
	x_LGN,y_LGN = 16,16
	x_cortex,y_cortex = 16,16
else:
	if scale==2:
		x_LGN,y_LGN = 64,64
		x_cortex,y_cortex = 64,64
		#x_LGN,y_LGN = 64,108
		#x_cortex,y_cortex = 64,108
	else:
		x_LGN,y_LGN = 32,32
		x_cortex,y_cortex = 32,32
		
if params['PBC']:
	assert x_LGN==x_cortex, "x_LGN!=x_cortex"
	assert y_LGN==y_cortex, "y_LGN!=y_cortex"


##paths
home_dir = os.environ["HOME"]
current_user = os.environ["USER"]
if current_user=="bettina":
	home_dir = home_dir + "/physics/columbia/projects"
if debug:
	data_dir = home_dir+'/miller94/data/debug/'
	image_dir = home_dir+'/miller94/image/debug/'
	movie_dir = home_dir+'/miller94/movie/debug/'
else:
	data_dir = home_dir+'/miller94/data/'
	image_dir = home_dir+'/miller94/image/'
	movie_dir = home_dir+'/miller94/movie/'
	
	
