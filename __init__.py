import os
import re
import h5py
from copy import copy

home_dir = os.environ["HOME"]
current_user = os.environ["USER"]


DA = 13 						#arbor diameter (default: 19 pixel)
N = 32

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
	
	

def write_to_hdf5(results_dict,version):
	filename = data_dir + "versions_v{}.hdf5".format(version)
	f = h5py.File(filename,'a')
	var_key_string = "{v}/".format(v=version)
	for key,value in results_dict.items():
		if (var_key_string in f.keys() and key in f[var_key_string].keys()):
			del f[var_key_string][key]
			f[var_key_string][key] = value
		else:
			f.create_dataset(var_key_string + "/" + key, data=value)
	f.close()
	print("Data written to versions_v{}.hdf5".format(version))


def get_version_id():
	file_list = os.listdir(data_dir)
	list_versions = [-1]
	for item in file_list:
		match = re.match("versions_v"+r'(\d+)', item)
		if match:
			list_versions.append(int(match.group(1)))
	max_present_version = max(list_versions)
	new_version = max_present_version + 1
	return new_version
	# filename = data_dir + "versions.hdf5"
	# if os.path.isfile(filename):
	# 	f = h5py.File(filename,'r')
	# 	previous_version_ids = list(f["version"].keys())
	# 	previous_version_ids = [int(item) for item in previous_version_ids]
	# 	new_id = max(previous_version_ids) + 1
	# 	f.close()
	# else:
	# 	new_id = 0
	return new_id

def load_hdf5(version):
	filename = data_dir + "versions_v{}.hdf5".format(version)
	if os.path.isfile(filename):
		f = h5py.File(filename,'r')
		data = {}
		for key in list(f[version].keys()):
			data[key] = f[version][key][()]
		f.close()
		return data
	else:
		print("File not found.")
		return None