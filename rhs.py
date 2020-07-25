import numpy as np
import warnings

#warnings.filterwarnings('error')

def synaptic_normalization_factor_init(s,arbor):
	'''multiplicative normalization of initial S weights over each
	cortical cell (step 5)'''
	return np.sum(arbor[:,:,0],axis=0)*2./np.sum(s[:,:,0]+s[:,:,1], axis=0)

def unconstrained_init(y,t,linit,I,arbor,C_onon,C_onoff,C_offoff,target_sigma):
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
	s_on = y[:,:,0]
	s_off = y[:,:,1]
	
	S_on = linit*np.dot(np.dot(C_onon,s_on) + np.dot(C_onoff,s_off),I)*arbor[:,:,0]
	S_off = linit*np.dot(np.dot(C_offoff,s_off) + np.dot(C_onoff,s_on),I)*arbor[:,:,1]
	l_new = target_sigma/np.std(np.dstack([S_on,S_off]))
	if l_new>linit:
		l_new = np.max([l_new/2.,linit])
	# l_new = 0.01
	return np.dstack([S_on,S_off]),l_new


def unconstrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff):
	"""
	input: y, time t, params
	y = [S_on,S_off]
	output: y'
	"""
	s_on = y[:,:,0]
	s_off = y[:,:,1]
	
	arbor_frzn = np.copy(arbor)
	try:
		arbor_frzn[frozen] = 0
	except Exception as e:
		print(e)
	delta_on = np.dot(np.dot(C_onon,s_on) + np.dot(C_onoff,s_off),I)*arbor_frzn[:,:,0]
	delta_off = np.dot(np.dot(C_offoff,s_off) + np.dot(C_onoff,s_on),I)*arbor_frzn[:,:,1]
	return np.dstack([delta_on,delta_off])


def constrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff):
	"""
	incorporates conservation of synaptic strength, and leads to competition
	"""
	arbor_actv = np.ma.array(arbor,mask=frozen)
	delta = unconstrained(y,t,frozen,I,arbor,C_onon,C_onoff,C_offoff)
	
	### set filled = 1 since otherwise if all synapses to one cortical\
	#neuron are frozen we get RuntimeWarning due to dividing by zero.
	### since a_frzn_on.filled(0)/a_frzn_off.filled(0) is only nonzero for\
	#active synapses, eps gets 0 nonetheless if all synapses to one cortical neuron are frozen.
	
	eps1 = 1.*np.sum(delta[:,:,0] + delta[:,:,1],axis=0)
	norm = arbor_actv[:,:,0].sum(axis=0).filled(0)+arbor_actv[:,:,1].sum(axis=0).filled(0)
	norm[norm==0] = 1
	try:
		eps = eps1/norm
	except RuntimeWarning:
		assert np.count_nonzero(eps1)==np.count_nonzero(norm), "np.count_nonzero(eps1)!=np.count_nonzero(norm)"
		norm[eps1==0] = 1
		eps = eps1/norm
	return np.dstack([delta[:,:,0] - eps[None,:]*arbor_actv[:,:,0].filled(0), delta[:,:,1] - eps[None,:]*arbor_actv[:,:,1].filled(0)])
	

def synaptic_normalization_factor(s,frozen,arbor):
	s_frzn = np.ma.array(s,mask=np.logical_not(frozen))
	s_actv = np.ma.array(s,mask=frozen)
	gamma =  (-s_frzn.sum(axis=0).sum(axis=1) + np.sum(arbor[:,:,0],axis=0)*2.)/(s_actv.sum(axis=0).sum(axis=1))
	#gamma =  ( np.sum(arbor[:,:,0],axis=0)*2.)/(s_actv.sum(axis=0).sum(axis=1))
	gamma_subject = np.clip(gamma.filled(1),0.8,1.2)
	return gamma_subject


def clip_synapse(s,s_max,arbor,frozen_old):
	smaller = s < 0
	larger = s > (s_max*arbor)
	frozen = np.logical_or(np.logical_or(larger, smaller), frozen_old)
	s[smaller] = 0
	s[larger] = arbor[larger]*s_max
	return s,frozen

