import numpy as np
import sys

import bettina.modeling.miller94.update_clean.rhs as rhs




def three_step_method(y,t0,l,f,frozen,I,arbor,C_onon,C_onoff,C_offoff,\
	s_max,c_orth,s_orth,mode_norm):
	'''integration scheme by Birkhoff and Rota 1978'''
	Nsyn = np.sum(arbor.flatten()>0)
	mode = mode_norm

	timesteps = 100
	frozen_ratio = np.zeros(timesteps)
	Nfrozens = np.sum(frozen[arbor>0])
	st = []
	
	## init
	y_0 = y
	st.append(y_0)
	frozen_ratio[0] = Nfrozens
		
	## first update step
	f_0 = f(y_0,t0,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth,s_orth,mode)
	y_1 = y_0 + l*f_0
	y_1,frozen = rhs.clip_synapse(y_1,s_max,arbor,frozen)
	if np.sum(frozen)>Nfrozens:
		y_1 *= rhs.synaptic_normalization_factor(y_1,frozen,arbor)
		Nfrozens = np.sum(frozen[arbor>0])
	st.append(y_1)
	frozen_ratio[1] = Nfrozens
	
	
	## second update step
	f_1 = f(y_1,1,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth,s_orth,mode)
	update_step = l*(2*f_1 - f_0)
	update_step[frozen] = 0.0
	y_2 = y_1 + update_step
	y_2,frozen = rhs.clip_synapse(y_2,s_max,arbor,frozen)
	if np.sum(frozen)>Nfrozens:
		y_2 *= rhs.synaptic_normalization_factor(y_2,frozen,arbor)
		Nfrozens = np.sum(frozen[arbor>0])
	st.append(y_2)
	frozen_ratio[2] = Nfrozens


	## third update step
	f_2 = f(y_2,2,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth,s_orth,mode)
	update_step = l*(23*f_2 - 16*f_1 + 5*f_0)/12.
	update_step[frozen] = 0.0
	y_3 = y_2 + update_step
	y_3,frozen = rhs.clip_synapse(y_3,s_max,arbor,frozen)
	if np.sum(frozen)>Nfrozens:
		y_3 *= rhs.synaptic_normalization_factor(y_3,frozen,arbor)
		Nfrozens = np.sum(frozen[arbor>0])
	st.append(y_3)
	frozen_ratio[3] = Nfrozens
	
	# ## fourth update step
	f_3 = f(y_3,3,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth,s_orth,mode)
	update_step = l*(23*f_3 - 16*f_2 + 5*f_1)/12.
	update_step[frozen] = 0.0
	y_4 = y_3 + update_step
	y_4,frozen = rhs.clip_synapse(y_4,s_max,arbor,frozen)
	if np.sum(frozen)>Nfrozens:
		y_4 *= rhs.synaptic_normalization_factor(y_4,frozen,arbor)
		Nfrozens = np.sum(frozen[arbor>0])
	st.append(y_4)
	frozen_ratio[4] = Nfrozens
	
	i = 0
	y_new = y_4
	frozen_limit = 0.96
	while (Nfrozens<(frozen_limit*Nsyn) and i<(timesteps-1)):	##original criterion for running time
	# for i in range(4,timesteps-1):
		print(i)
		sys.stdout.flush()
		
		f_4 = f(y_4,i,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth,s_orth,mode)
		update_step = 2*l*(23*f_4 - 16*f_2 + 5*f_0)/12.	
		update_step[frozen] = 0.0
		y_new = y_4 + update_step
		y_new,frozen = rhs.clip_synapse(y_new,s_max,arbor,frozen)
		if np.sum(frozen)>Nfrozens:
			y_new *= rhs.synaptic_normalization_factor(y_new,frozen,arbor)
			Nfrozens = np.sum(frozen[arbor>0])
		# st.append(y_new)
		frozen_ratio[i+1] = 1.*Nfrozens
		
		f_0 = f_2
		f_2 = f_4
		y_4 = y_new
		

		if (frozen_ratio[i+1]/Nsyn)>frozen_limit:
			## st = st[...,:i+2]
			frozen_ratio = frozen_ratio[:i+2]
			break
		i += 1

		
	st.append(y_new)
	#print('st on',np.sum(st[:1,:,0,:],axis=1))
	#print('st of',np.sum(st[:1,:,1,:],axis=1))
	print(1.*Nfrozens/Nsyn,frozen_ratio[-10:]/Nsyn,Nsyn)
	frozen_ratio = 1.*frozen_ratio/Nsyn
	# st = np.array(st)
	# num_timesteps = st.shape[0]
	# st = st[[0,num_timesteps//4,-1]]
	return y_new,np.array(st),frozen_ratio,frozen
	
	
