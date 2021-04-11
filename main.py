#!/usr/bin/python
import numpy as np

import matplotlib.pyplot as plt

if __name__=="__main__":
	from tools import parse_args,tools,plotting,network_system,params,three_step_method,\
	N,DA,interaction_params,LGN_corr_params,rhs

	mode_norm = "x"
	c_orth,s_orth  = None,None

	version = 28#tools.get_version_id()
	print("version",version)
	rA = params["rA"]
	rI = parse_args.args.intracortical_scale * rA
	rC = parse_args.args.LGNinput_scale * rA

	
	N = params["N"]
	profile = "Gaussian"
	
	system = network_system.Network((N,N),(N,N),version)

	connectivity_params = {"ampl" : interaction_params["ampl"], "sigma" : rI}
	xI = np.min([2*rI,N//2])
	i_fct = system.create_matrix(connectivity_params,interaction_params["profile"],rA=xI)

	corr_params = {"ampl" : LGN_corr_params["ampl"], "sigma" : rC}
	c_onon = system.create_matrix(corr_params,LGN_corr_params["profile"])
	c_offoff = np.copy(c_onon)
	corr_params = {"ampl" : -0.5*LGN_corr_params["ampl"], "sigma" : rC}
	c_onoff = system.create_matrix(corr_params,LGN_corr_params["profile"])

	arbor_sgl = system.create_arbor(radius=rA,profile="heaviside")
	arbor = np.dstack([arbor_sgl,arbor_sgl])
	print("SHAPE",arbor.shape,c_onon.shape,i_fct.shape)

	# ##################################################################################
	# idx = N//2
	# ncol,nrow = 5,2
	# fig=plt.figure(figsize=(5*6,4))
	# ax = fig.add_subplot(nrow,ncol,1)
	# ax.set_title("arbor")
	# im=ax.imshow(arbor.reshape(N,N,N,N,2)[idx,idx,:,:,0],interpolation='nearest')
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(nrow,ncol,2)
	# ax.set_title("C_onon")
	# im2=ax.imshow(c_onon.reshape(N,N,N,N)[idx,idx,:,:],interpolation='nearest',\
	# 				cmap="binary")
	# plt.colorbar(im2,ax=ax)
	# ax = fig.add_subplot(nrow,ncol,3)
	# ax.set_title("C_offoff")
	# im2=ax.imshow(c_offoff.reshape(N,N,N,N)[idx,idx,:,:],interpolation='nearest',\
	# 				cmap="binary")
	# plt.colorbar(im2,ax=ax)
	# ax = fig.add_subplot(nrow,ncol,4)
	# ax.set_title("I")
	# im2=ax.imshow(i_fct.reshape(N,N,N,N)[idx,idx,:,:],interpolation='nearest',\
	# 				cmap="binary")
	# plt.colorbar(im2,ax=ax)
	# ax = fig.add_subplot(nrow,ncol,5)
	# ax.set_title("C_D")
	# im2=ax.imshow((c_onon-c_onoff).reshape(N,N,N,N)[idx,idx,:,:],\
	# 				interpolation='nearest',cmap="binary")
	# plt.colorbar(im2,ax=ax)
	# ax = fig.add_subplot(nrow,ncol,ncol+1)
	# ax.plot(arbor.reshape(N,N,N,N,2)[idx,idx,idx,:,0],"-+")
	# ax = fig.add_subplot(nrow,ncol,ncol+2)
	# ax.plot(c_onon.reshape(N,N,N,N)[idx,idx,idx,:],"-+")
	# ax = fig.add_subplot(nrow,ncol,ncol+3)
	# ax.plot(c_offoff.reshape(N,N,N,N)[idx,idx,idx,:],"-+")
	# ax = fig.add_subplot(nrow,ncol,ncol+4)
	# ax.plot(i_fct.reshape(N,N,N,N)[idx,idx,idx,:],"-+")
	# ax = fig.add_subplot(nrow,ncol,ncol+5)
	# ax.plot((c_onon-c_onoff).reshape(N,N,N,N)[idx,idx,idx,:],"-+")
	# plt.show()
	# exit()
	###################################################################################3


	# np.random.seed(20191218)
	# s_init = arbor*(1 + params["s_noise"]*np.random.choice([-1,1],(N*N,N*N,2)))
	# s_init = arbor * (1 + params["s_noise"] * system.rng.choice([-1,1],(N*N,N*N,2)))
	s_init = tools.create_RFs("initialize",s_noise=params["s_noise"],arbor=arbor,N=N)
	# s_init = tools.create_RFs("gabor",N=N,DA=DA)

	s_init *= rhs.synaptic_normalization_factor_init(s_init,arbor)
	# print("s_init",np.nanmax(s_init),np.nanmin(s_init))
	# exit()
	## step 5 (see Miller 1994)
	s_init,frozen_init = rhs.clip_synapse(s_init,params["s_max"],arbor,\
							np.zeros_like(s_init,dtype=bool))
	## step 1 (see Miller 1994)
	delta,lambda_normalised = rhs.unconstrained_init(s_init,0,params["lambda_init"],i_fct,\
								arbor,c_onon,c_onoff,c_offoff,params["target_sigma"])

	lambda_normalised = lambda_normalised# * 0.05
	print("lambda_normalised",lambda_normalised)
	## step 2 (see Miller 1994)
	s1,st,frozen_ratio,frozen = three_step_method.three_step_method(s_init,\
																0,\
																lambda_normalised,\
																rhs.constrained,\
																frozen_init,\
																i_fct,\
																arbor,\
																c_onon,\
																c_onoff,\
																c_offoff,\
																params["s_max"],\
																c_orth,s_orth,\
																mode_norm)
	print('frozen ratio at {}.'.format(frozen_ratio[-1]))
	print("s1",s1.shape,st.shape)

	# fig = plt.figure()
	# ax = fig.add_subplot(141)
	# im=ax.imshow(s1[:,:,0],interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(142)
	# im=ax.imshow(s1[0,:,0].reshape(N,N),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(143)
	# im=ax.imshow(s1[0,:,1].reshape(N,N),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# ax = fig.add_subplot(144)
	# im=ax.imshow(s1[310,:,0].reshape(N,N),interpolation="nearest",cmap="binary")
	# plt.colorbar(im,ax=ax)
	# plt.show()
	# exit()

	##do plotting
	# plotting.plot_figures(st,N=N,version=version,DA=params["DA"]+2,arbor=arbor_sgl)

	results_dict = {"rC" : rC, "rI" :  rI, "s" : s1, "rA" : rA, "N" : N}
	# tools.write_to_hdf5(results_dict,version)

	print("done")