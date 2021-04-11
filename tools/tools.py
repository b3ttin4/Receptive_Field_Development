import numpy as np
import re
import os
import h5py

from tools import data_dir,network_system

def gen_ecp(x, y, conn_params):
	if "kc" in conn_params.keys():
		kc = conn_params["kc"]
	else:
		kc = 3.## number of wavelengths per layer
	if "n" in conn_params.keys():
		n = conn_params["n"]
	else:
		n = 30
	rng = conn_params["rng"]
	A = (rng.random(n)*0.2+0.8)[:,None,None]#1.0

	## Long-range interactions
	j = np.arange(n)
	kj = kc*2*np.pi*np.stack([np.cos(j*np.pi/n), np.sin(j*np.pi/n)])
	lj = rng.choice([-1.,1.],size=n)
	lk = (lj[None,:]*kj)[:,:,None,None]
	phi = rng.uniform(low=0,high=2*np.pi,size=n)[:,None,None]
	ecp = np.sum(A*np.exp(1j*lk[0,...]*x[None,...] + 1j*lk[1,:,...]*y[None,...] + phi),axis=0)
	sigma = 0.3/kc/2/np.pi*n
	return ecp, sigma


def create_RFs(mode,**kwargs):
	if mode=="initialize":
		arbor = kwargs["arbor"]
		s_noise = kwargs["s_noise"]
		N = kwargs["N"]
		if "rng" in kwargs.keys():
			rng = kwargs["rng"]
		else:
			rng = np.random.RandomState(20210311)
		s_full = arbor*(1 + s_noise*np.random.choice([-1,1],(N*N,N*N,2)))

	elif mode=="gabor":
		N = kwargs["N"]
		DA = kwargs["DA"]

		## smooth OPM generation
		grid = np.linspace(0,1,N,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200205)}
		ecp,sigma = gen_ecp(xto, yto, conn_params)
		opm = 0*np.angle(ecp,deg=False)*0.5
		# opm[0,0] = np.pi/2.
		## smooth phases generation
		grid = np.linspace(0,1,N,endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		conn_params = {"rng" : np.random.RandomState(20200206), "kc" : 1.5, "n" : 1}
		ecp,sigma = gen_ecp(xto, yto, conn_params)
		pref_phase = 0*np.angle(ecp,deg=False)
		
		network = network_system.Network((N,N),(N,N),1)
		conn_params = {"sigma" : 0.2,
						"ampl" : 1.,
						"theta" : opm,#0.3*np.ones((Nlgn,Nlgn)),
						"psi" : pref_phase,
						"freq" : 10}
		gb = network.create_matrix(conn_params, "Gabor")
		s_on = np.copy(gb)
		s_off = np.copy(gb)
		s_on[s_on<0] = 0
		s_off[s_off>0] = 0
		s_off *= -1.
		s_full = np.dstack([s_on,s_off])

	return s_full


def get_response(sd,DA):
	assert sd.ndim==4, "reshape sd such that it is four dimensional"
	N4 = sd.shape[0]
	Nlgn = sd.shape[2]

	## Fourier transform SD
	fct = 1
	delta_bins = 20
	sdfft = np.abs(np.fft.fftshift(np.fft.fftn(sd,s=(fct*Nlgn,fct*Nlgn),axes=(2,3)),axes=(2,3)))
	sdfft2 = sdfft[:,:,:sdfft.shape[2]//2,:]
	h,w = sdfft2.shape[2:]
	sdfft_long = sdfft2.reshape(N4,N4,-1)

	## bin orientation in fourier space
	kx,ky = np.meshgrid(np.arange(-w/2.,w/2.),np.arange(0,h)[::-1])
	angles = np.arctan2(ky,kx)*180/np.pi + (np.arctan2(ky,kx)<0)*360
	frequency = np.sqrt((kx/w)**2 + (ky/h)**2).flatten()
	angle_disc = np.arange(0,180,delta_bins)
	ori_bins = np.searchsorted(angle_disc,angles,side='right')
	ori_bins = ori_bins.flatten()

	## best response for each binned orientation across spatial frequency
	Rn = np.empty((180//delta_bins,N4,N4))*np.nan
	maxk = np.zeros((180//delta_bins,N4,N4),dtype=int)
	for ibin in range(1,1+180//delta_bins):
		sd_k = sdfft_long[:,:,ori_bins==ibin]
		sd_maxk = np.argmax(sd_k,axis=2)
		Rn[ibin-1,:,:] = np.max(sd_k,axis=2)
		maxk[ibin-1,:,:] = sd_maxk

	## vector sum of best responses
	phi = np.linspace(0,2*np.pi,num=180//delta_bins,endpoint=False) + delta_bins/2./180*np.pi
	opm = np.sum(np.exp(1j*phi)[:,None,None]*Rn, axis=0)
	rms = np.sum(Rn,axis=0)
	opm = opm/rms
	return opm,Rn

def get_RF(s,DA):
	N = s.shape[0]
	RF = np.zeros((3,DA*N,DA*N))
	PF = np.zeros((3,DA*N,DA*N))
	for i in range(N):
		for j in range(N):
			son_ij = np.roll(np.roll(s[:,:,j,i,0],shift=N//2-j,axis=0),shift=N//2-i,axis=1)
			sof_ij = np.roll(np.roll(s[:,:,j,i,1],shift=N//2-j,axis=0),shift=N//2-i,axis=1)
			# print(i,j,son_ij.shape,sof_ij.shape,N//2-DA//2,N//2+DA//2+DA%2)
			RF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 son_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2] -\
			 sof_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
			RF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 son_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
			RF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 sof_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
			
	for i in range(N):
		for j in range(N):
			son_ij = np.roll(np.roll(s[j,i,:,:,0],shift=N//2-j,axis=0),shift=N//2-i,axis=1)
			sof_ij = np.roll(np.roll(s[j,i,:,:,1],shift=N//2-j,axis=0),shift=N//2-i,axis=1)
			PF[0,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 son_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2] -\
			 sof_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
			PF[1,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 son_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
			PF[2,j*DA:(j+1)*DA,i*DA:(i+1)*DA] = \
			 sof_ij[N//2-DA//2:N//2+DA//2+DA%2, N//2-DA//2:N//2+DA//2+DA%2]
	return RF,PF


def print_used_simulation_params():
	file_list = os.listdir(data_dir)
	params = []
	for item in file_list:
		match = re.match("versions_v"+r'(\d+)', item)
		if match:
			version = match.group(1)
			data = load_hdf5(version)
			if "rA" in data.keys():
				rA = data["rA"][()]
			else:
				rA = 6.5
			rI = data["rI"][()]/rA
			rC = data["rC"][()]/rA
			params.append(np.array([float(version),rA,rI,rC]))
	return params


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

