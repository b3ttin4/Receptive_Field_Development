#!/usr/bin/python
import numpy as np

from tools import tools,params,three_step_method,N,DA,rhs


def run_simulation(rA,rI,rC):
	# s_init = arbor*(1 + params["s_noise"]*np.random.choice([-1,1],(N*N,N*N,2)))
	# s_init = arbor * (1 + params["s_noise"] * system.rng.choice([-1,1],(N*N,N*N,2)))
	s_init = tools.create_RFs("initialize",s_noise=params["s_noise"],arbor=arbor,N=N,\
							  rng=np.random.RandomState(2021))#20210311
	# s_init = tools.create_RFs("gabor",N=N,DA=DA)
	## 20210: perfect R
	## 202103: almost perfect R, but on/off different sizes

	s_init *= rhs.synaptic_normalization_factor_init(s_init,arbor,mode_norm)
	## step 5 (see Miller 1994)
	s_init,frozen_init = rhs.clip_synapse(s_init,params["s_max"],arbor,\
											np.zeros_like(s_init,dtype=bool))
	## step 1 (see Miller 1994)
	delta,lambda_normalised = rhs.unconstrained_init(s_init,0,params["lambda_init"],i_fct,\
														arbor,c_onon,c_onoff,c_offoff,\
														params["target_sigma"],mode_norm)

	lambda_normalised = lambda_normalised
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
	return s1



if __name__=="__main__":
	import matplotlib.pyplot as plt
	from tools import parse_args,network_system,interaction_params,LGN_corr_params

	mode_norm = "x"
	c_orth,s_orth  = None,None

	version = tools.get_version_id()
	print("version",version)
	rA = params["rA"]
	rI = parse_args.args.intracortical_scale * rA
	rC = parse_args.args.LGNinput_scale * rA

	print(rA,rI,rC)

	
	N = params["N"]
	profile = "Gaussian"
	
	system = network_system.Network((N,N),(N,N),version)

	## Gaussian recurrent connectivity
	connectivity_params = {"ampl" : interaction_params["ampl"], "sigma" : rI}
	xI = np.min([2*rI,N//2])
	i_fct = system.create_matrix(connectivity_params,interaction_params["profile"],rA=xI)
	## MH recurrent connectivity
	# connectivity_params = {"ampl" : interaction_params["ampl"], "sigma1" : rI, "sigma2" : rI*2}
	# xI = np.min([2*rI,N//2])
	# i_fct = system.create_matrix(connectivity_params,"Mexican-hat",rA=xI)


	corr_params = {"ampl" : LGN_corr_params["ampl"], "sigma" : rC}
	c_onon = system.create_matrix(corr_params,LGN_corr_params["profile"])
	c_offoff = np.copy(c_onon)
	corr_params = {"ampl" : -0.5*LGN_corr_params["ampl"], "sigma" : rC}
	c_onoff = system.create_matrix(corr_params,LGN_corr_params["profile"])

	arbor_sgl = system.create_arbor(radius=rA,profile="gaussian")
	arbor = np.dstack([arbor_sgl,arbor_sgl])
	print("SHAPE",arbor.shape,c_onon.shape,i_fct.shape)


	s1 = run_simulation(rA,rI,rC)
	results_dict = {"rC" : rC, "rI" :  rI, "s" : s1, "rA" : rA, "N" : N}
	tools.write_to_hdf5(results_dict,version)

	print("done")