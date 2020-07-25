import numpy as np
from bettina.modeling.miller94.conf import params
from bettina.tools import gen_gaussian_random_field as gen_grf

DA = params['DA']
rA = params['rA']
cA = params['cA']
gammaC = params['gammaC']
gammaI = params['gammaI']
aI = params['aI']
PBC = params['PBC']


def distance(x0, x1, dimensions):
	''' PBC: periodic boundary conditions'''
	if PBC:
		delta = x0 - x1
		signs = np.sign(delta)
		return np.where(np.abs(delta) > (0.5 * dimensions), -signs*(dimensions - np.abs(delta)), delta)
	else:
		return x0 - x1

def gaussian(x,y,s):
	return np.exp(-(x**2+y**2)/s**2)


def Arbor(dx,dy):
	"""
	arbor function: crude model of overlap of geniculocortical terminal
	arbors with cortical dendritic arbors
	dx,dy: difference between 2-dim coordinates in LGN and cortex
	"""
	d = np.sqrt(dx**2+dy**2)		## absolute distance |vec(x)-vec(y)|
	d_unique = np.unique(d[d<=(DA/2.)])
	N = rA*(cA+2)
	
	delta_bin = 0.05
	i,j = np.meshgrid(np.arange(-(N//2),N//2+delta_bin,delta_bin),np.arange(-(N//2),N//2+delta_bin,delta_bin))
	delta1 = np.sqrt(i**2+j**2)
	delta2 = np.sqrt((d_unique[:,None,None]-i[None,:,:])**2+(j[None,:,:])**2)
	circles = (delta1 < rA)[None,:,:]*(delta2 < (rA*cA))
	norm = np.sum(delta2 < (rA*cA),axis=(1,2))
	overlap = 1.*np.sum(circles,axis=(1,2))/norm
	
	normalized = np.zeros_like(d)
	for iduni in range(d_unique.size):
		normalized[d==d_unique[iduni]] = overlap[iduni]
	return np.dstack([normalized,normalized])


def C_onon(x,y,rc):
	"""
	C: afferent correlation function
	given by a difference of Gaussians
	x,y: coordinates in LGN
	rc: determines width of correlation
	"""
	return gaussian(x,y,rc*(DA/2.)) - 1./gammaC**2*gaussian(x,y,gammaC*rc*(DA/2.))
	
def C_offoff(x,y,rc):
	return C_onon(x,y,rc)

def C_onoff(x,y,rc):
	return -0.5*C_onon(x,y,rc)

def C_D(x,y,rc):
	return C_onon(x,y,rc) - C_onoff(x,y,rc)

def a(x,y):
	"""
	reduces size of intercell interactions relative to interactions between synapses on same postsynaptic cell
	"""
	return ((x==0)*(y==0)).astype(float) + np.logical_not((x==0)*(y==0)).astype(float)*aI


def I(x,y,rI,mode,*params):
	"""
	I: intracortical interaction function
	x,y: coordinates in cortex
	rI: systematically varied, controls peak of FT(I)/width of I
	mode: determines interaction type
	params: additional parameters for mode='data'
	"""
	if mode=='exc':
		return gaussian(x,y,rI*6.5)*a(x,y)
	elif mode=='inh2':
		return a(x,y)*(gaussian(x,y,rI*6.5) - 0.5*gaussian(x,y,gammaI*rI*6.5))
	elif mode=='inh':
		return a(x,y)*(gaussian(x,y,rI*6.5) - 1./gammaC**2*gaussian(x,y,gammaI*rI*6.5))
	elif mode=='const':
		return np.diagflat(np.ones(x.shape[0]))
	elif mode=='Rgauss':
		ring_size = 3
		ring_thickness = 1
		oris,sel = gen_grf.generate_topology_map(params[1],params[0], ring_size,\
		 ring_thickness, rng=np.random.RandomState(1), return_complex=False)
		oris = oris.reshape(params[0]*params[1])
		sel = sel.reshape(params[0]*params[1])
		return np.sqrt(sel[:,None]*sel[None,:])*np.cos(2*(oris[:,None]-oris[None,:])),oris.reshape(params[1],params[0]),sel.reshape(params[1],params[0])
	elif mode=='Rgauss2':
		ring_size = 3
		ring_thickness = 1
		oris,sel = gen_grf.generate_topology_map(params[1],params[0], ring_size,\
		 ring_thickness, rng=np.random.RandomState(1), return_complex=False)
		oris = oris.reshape(params[0]*params[1])
		return np.cos(2*(oris[:,None]-oris[None,:])),oris.reshape(params[1],params[0]),sel
	elif mode=='pw':
		ky,kx = 1.,1.
		return np.cos(2*np.pi*(ky/(params[1]//2)*y+kx/(params[0]//2)*x))
	elif mode=='data':
		from bettina.modeling.tools import load_routine
		h,w = params[4:6]
		ferret,date,tseries = params[:3]
		VERSION = params[3]
		roi,scale = load_routine.load_roi(ferret,date,tseries)
		korr_matrix = load_routine.load_korr(ferret,date,tseries,VERSION)
		section = np.zeros_like(roi)
		if scale==2:
			section[20:20+w,21:21+h] = True*roi[20:20+w,21:21+h]
			#section[10:10+w,21:21+h] = True*roi[10:10+w,21:21+h]
			print('CHECK',h*w,np.sum(section))
			#exit()
		else:
			section[5:5+w,11:11+h] = True*roi[5:5+w,11:11+h]
			print('CHECK',h*w,np.sum(section))
			##section[35:35+w,35:35+h] = True*roi[35:35+w,35:35+h]
		part = (korr_matrix[section[roi],:].T)[section[roi],:]
		return part
	else:
		raise ValueError('Wrong mode in fct I(x,y,rI,mode,*params). Use one of: exc, inh, const, data!')

	
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	
	x_LGN,y_LGN = 12,12
	x_crtx,y_crtx = 12,12

	x,y = np.meshgrid(np.arange(0,x_LGN,1),np.arange(0,y_LGN,1))
	coord_LGN = np.dstack([x,y]).reshape(x_LGN*y_LGN,2)

	x,y = np.meshgrid(np.arange(0,x_crtx,1),np.arange(0,y_crtx,1))
	coord_crtx = np.dstack([x,y]).reshape(x_crtx*y_crtx,2)

	x_delta_LGN_crtx = distance(coord_LGN[:,0,None], coord_crtx[None,:,0],x_LGN)
	y_delta_LGN_crtx = distance(coord_LGN[:,1,None], coord_crtx[None,:,1],y_LGN)

	arbor = Arbor(x_delta_LGN_crtx,y_delta_LGN_crtx)
	
	plt.imshow(arbor[:,17,0].reshape(12,12),interpolation="nearest",cmap="binary")
	plt.colorbar()
	plt.show()
	
	#x,y = np.meshgrid(np.arange(0,Nx,1),np.arange(0,Ny,1))
	#coord = np.dstack([x,y]).reshape(Nx*Ny,2)
	
	#x_delta = coord[:,0,None] - coord[None,:,0]
	#y_delta = coord[:,1,None] - coord[None,:,1]
	
	##N = 3
	##x=np.arange(-N,N,1)
	#x,y = np.meshgrid(np.arange(0,N,1),np.arange(0,N,1))
	#coord = np.dstack([x,y]).reshape(N*N,2)
	#print((coord[:,0,None] - coord[None,:,0]))
	##i = I(x,y,0.2,'Rgauss',64,64)
	##print(i)
	
	#print(np.sum(C_onon(x,y,0.3)))
	
	#plt.figure()
	#plt.imshow(C_onon(x,y,0.3),interpolation='nearest')
	#plt.colorbar()
	#plt.show()
	
	
