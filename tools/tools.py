import numpy as np
import re
import os

from bettina.modeling.miller94.update_clean import network_system,load_hdf5,data_dir


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



if __name__=="__main__":
	import matplotlib.pyplot as plt
	from bettina.modeling.miller94.update_clean.plotting import plotting
	from bettina.modeling.miller94.update_clean import N,DA,params,rhs

	system = network_system.Network((N,N),(N,N),1)

	arbor_sgl = system.create_arbor(radius=params["rA"],profile="gaussian")
	arbor = np.dstack([arbor_sgl,arbor_sgl])

	# s_full = create_RFs("initialize",s_noise=params["s_noise"],arbor=arbor,N=N)
	s_full = create_RFs("gabor",N=N,DA=DA)
	s_full *= rhs.synaptic_normalization_factor_init(s_full,arbor)
	print("s_full",s_full.shape)
	s_full = s_full.reshape(N,N,N,N,2)
	# s_full[s_full>0] = 1
	print("s_full",s_full.shape,N,DA)
	RF,_ = plotting.get_RF(s_full,DA)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title("S_D")
	im = ax.imshow(RF[1,:,:],interpolation='nearest',cmap='RdBu_r')
	plt.colorbar(im,ax=ax)
	plt.show()