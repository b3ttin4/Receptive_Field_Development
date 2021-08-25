import numpy as np
import warnings

#warnings.filterwarnings('error')

def synaptic_normalization_factor_init(s,arbor,mode):
	'''multiplicative normalization of initial S weights over each
	cortical cell (step 5)'''
	if mode=="x":
		factor_on = np.sum(arbor[:,:,0],axis=1)/np.sum(s[:,:,0],axis=1)
		factor_of = np.sum(arbor[:,:,1],axis=1)/np.sum(s[:,:,1],axis=1)
		f = np.dstack([ factor_on[:,None],factor_of[:,None] ])
	elif mode=="alpha":
		f = (np.sum(arbor[:,:,0],axis=0)*2./np.sum(s[:,:,0]+s[:,:,1], axis=0))[None,:,None]
	return f

def unconstrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff):
	"""
	equation (1) in Miller 1994
	input: y, time t, params
	y = [S_on,S_off]
	output: deltay (which is zero for frozen weights)
	"""
	s_on = y[:,:,0]
	s_off = y[:,:,1]
	
	arbor_frzn = np.copy(arbor)
	# arbor_frzn[frozen] = 0

	delta_on = np.dot(np.dot(C_onon,s_on) + np.dot(C_onoff,s_off),I) * arbor_frzn[:,:,0]
	delta_off = np.dot(np.dot(C_offoff,s_off) + np.dot(C_onoff,s_on),I) * arbor_frzn[:,:,1]
	return np.dstack([delta_on,delta_off])

def unconstrained_init(y,t,linit,I,arbor,C_onon,C_onoff,C_offoff,target_sigma,mode):
	"""
	input: y, time t, params
		y = [S_on,S_off]
		linit : initial lambda (growth constant)
		I : rec interaction fct
		arbor : extension of ff weights from one LGN unit
		C_onon/C_onoff/C_offoff : correlation between on and on/off LGN units
		target_sigma : target value for std of change in s weight
	output: y'
	"""
	frozen = np.zeros_like(arbor,dtype=bool)
	deltaS_unconstrained = unconstrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff)
	
	l_new = target_sigma/np.std(linit * deltaS_unconstrained)
	if l_new>linit:
		l_new = np.max([l_new/2.,linit])
	return deltaS_unconstrained,l_new


def constrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff,c_orth=None,s_orth=None,mode="x"):
	"""
	incorporates conservation of synaptic strength, and leads to competition
	"""
	arbor_act = np.copy(arbor)
	# arbor_act[frozen] = 0.0
	delta = unconstrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff)
	
	### set filled = 1 since otherwise if all synapses to one cortical\
	#neuron are frozen we get RuntimeWarning due to dividing by zero.
	### since a_frzn_on.filled(0)/a_frzn_off.filled(0) is only nonzero for\
	#active synapses, eps gets 0 nonetheless if all synapses to one cortical neuron are frozen.
	if mode=="x":
		## motivated by equations (2,3) in Miller 1994
		## sum over x
		norm_on = arbor_act[:,:,0].sum(axis=1)
		norm_on[norm_on==0] = 1
		eps_on = 1.*np.sum(delta[:,:,0],axis=1)/norm_on 
		
		norm_of = arbor_act[:,:,1].sum(axis=1)
		norm_of[norm_of==0] = 1
		eps_of = 1.*np.sum(delta[:,:,1],axis=1)/norm_of  

		delta2 = np.dstack([delta[:,:,0] - eps_on[:,None]*arbor_act[:,:,0],\
				 			delta[:,:,1] - eps_of[:,None]*arbor_act[:,:,1]])

	elif mode=="alpha":
		## equations (2,3) in Miller 1994
		## sum over alpha and on/off
		norm = arbor_act[:,:,0].sum(axis=0) + arbor_act[:,:,1].sum(axis=0)
		norm[norm==0] = 1
		eps = (1.*np.sum(delta[:,:,0] + delta[:,:,1],axis=0)) / norm		
		delta2 = np.dstack([delta[:,:,0] - eps[None,:]*arbor_act[:,:,0],\
						    delta[:,:,1] - eps[None,:]*arbor_act[:,:,1]])
	
	elif mode=="xalpha":
		if True:
			## sum over alpha and on/off
			norm = arbor_act[:,:,0].sum(axis=0)+arbor_act[:,:,1].sum(axis=0)
			norm[norm==0] = 1
			eps = (1.*np.sum(delta[:,:,0] + delta[:,:,1],axis=0)) / norm		
			delta = np.dstack([delta[:,:,0] - eps[None,:]*arbor_act[:,:,0],\
			 delta[:,:,1] - eps[None,:]*arbor_act[:,:,1]])

			## sum over x
			norm_on = arbor_act[:,:,0].sum(axis=1)
			norm_on[norm_on==0] = 1
			eps_on = 1.*np.sum(delta[:,:,0],axis=1)/norm_on 
			
			norm_of = arbor_act[:,:,1].sum(axis=1)
			norm_of[norm_of==0] = 1
			eps_of = 1.*np.sum(delta[:,:,1],axis=1)/norm_of  

			delta2 = np.dstack([delta[:,:,0] - eps_on[:,None]*arbor_act[:,:,0],\
			 delta[:,:,1] - eps_of[:,None]*arbor_act[:,:,1]])		
			print("delta",t,np.nanmin(delta),np.nanmax(delta))
			print("delta2",t,np.nanmin(delta2),np.nanmax(delta2))

		else:
			arbor2 = np.swapaxes(np.swapaxes(arbor,0,2),1,2)
			# frozen = np.swapaxes(np.swapaxes(frozen,0,2),1,2)
			delta = np.swapaxes(np.swapaxes(delta,0,2),1,2)

			delta_mask = delta[arbor2>0]
			# mask_fl = arbor2[np.logical_and(arbor2>0,np.logical_not(frozen))]

			delta_mask -= np.sum(s_orth * np.dot(c_orth,delta_mask)[:,None],axis=0)
			# delta_mask *= mask_fl
			print("delta_mask",t,np.nanmin(delta),np.nanmax(delta))
			print("delta_mask",t,np.nanmin(delta_mask),np.nanmax(delta_mask))
			# delta2 = np.zeros_like(delta)
			delta[arbor2>0] = delta_mask
			# delta2 = delta2.reshape(delta.shape)

			# frozen = np.swapaxes(np.swapaxes(frozen,0,2),1,2)
			delta2 = np.swapaxes(np.swapaxes(delta,0,2),0,1)
			print("delta2",np.sum(np.sum(delta2,axis=1)>0.0000001),\
				np.sum(np.sum(delta2,axis=(0,2))>0.0000001))
	return delta2
	

def synaptic_normalization_factor(s,frozen,arbor,mode):
	if mode=="x":
		s_frzn = np.copy(s)
		s_actv = np.copy(s)
		s_frzn[np.logical_not(frozen)] = 0.0
		s_actv[frozen] = 0.0

		gamma = np.ones_like(arbor,dtype=float)
		factor_on = np.sum(-s_frzn[:,:,0] + arbor[:,:,0],axis=1)/np.sum(s_actv[:,:,0],axis=1)
		factor_of = np.sum(-s_frzn[:,:,1] + arbor[:,:,1],axis=1)/np.sum(s_actv[:,:,1],axis=1)
		gamma[...] = np.dstack([ factor_on[:,None],factor_of[:,None] ])
		gamma[frozen] = 1.
		gamma[np.logical_not(arbor)] = 1.
		gamma = np.clip(gamma,0.8,1.2)
	elif mode=="alpha":
		s_frzn = np.copy(s)
		s_actv = np.copy(s)
		s_frzn[np.logical_not(frozen)] = 0.0
		s_actv[frozen] = 0.0

		gamma = np.ones_like(arbor,dtype=float)
		gamma[...] =\
		  ((-s_frzn.sum(axis=0).sum(axis=1) + np.sum(arbor[:,:,0],axis=0)*2.)/\
		  	(s_actv.sum(axis=0).sum(axis=1)))[None,:,None]
		gamma = np.clip(gamma,0.8,1.2)
		gamma[frozen] = 1.
		gamma[np.logical_not(arbor)] = 1.
	return gamma


def clip_synapse(s,s_max,arbor,frozen_old):
	smaller = s <= 0
	larger = s >= (s_max*arbor)
	# frozen = np.logical_or(np.logical_or(larger, smaller), frozen_old)
	frozen = np.logical_or(larger, smaller)
	s[smaller] = 0
	s[larger] = arbor[larger]*s_max
	return s,frozen

