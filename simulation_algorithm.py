#!/usr/bin/python
import numpy as np
from bettina.modeling.miller94.functions import I,Arbor,C_onon,C_onoff,C_offoff,distance
import bettina.modeling.miller94.conf as conf
from bettina.tools import ensure_path

from bettina.modeling.miller94 import rhs as rhs
from bettina.modeling.tools import three_step_method as three_step_method

#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt


## VARYING PARAMETERS
rI = 0.3#0.8*conf.params["rA"]/np.sqrt(2.)/6.5#
rc = 0.268# 1.2/np.sqrt(2.)#
t0 = 0
mode = 'inh'
show_intermediate_results = False



PBC = conf.params['PBC']
if PBC:
	npbc=''
else:
	npbc = '_npbc'

## DEFINE SYSTEM
DA = conf.params['DA']
target_sigma = conf.params['target_sigma']
s_max = conf.params['s_max']
s_noise = conf.params['s_noise']
lambda_init = conf.params['lambda_init']

x_LGN,y_LGN = conf.x_LGN,conf.y_LGN
x_crtx,y_crtx = conf.x_cortex,conf.y_cortex

x,y = np.meshgrid(np.arange(0,x_LGN,1),np.arange(0,y_LGN,1))
coord_LGN = np.dstack([x,y]).reshape(x_LGN*y_LGN,2)

x,y = np.meshgrid(np.arange(0,x_crtx,1),np.arange(0,y_crtx,1))
coord_crtx = np.dstack([x,y]).reshape(x_crtx*y_crtx,2)

x_delta_LGN_LGN = distance(coord_LGN[:,0,None], coord_LGN[None,:,0],x_LGN)
y_delta_LGN_LGN = distance(coord_LGN[:,1,None], coord_LGN[None,:,1],y_LGN)

x_delta_LGN_crtx = distance(coord_LGN[:,0,None], coord_crtx[None,:,0],x_LGN)
y_delta_LGN_crtx = distance(coord_LGN[:,1,None], coord_crtx[None,:,1],y_LGN)

x_delta_crtx_crtx = distance(coord_crtx[:,0,None], coord_crtx[None,:,0], x_crtx)
y_delta_crtx_crtx = distance(coord_crtx[:,1,None], coord_crtx[None,:,1], y_crtx)


if mode=='data':
	ferret = 1581
	date = '2013-12-20'
	tseries = 1
	VERSION = 1
	i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,ferret,date,tseries,VERSION,x_crtx,y_crtx)
elif mode in ('Rgauss','Rgauss2'):
	i,oris,sel = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,x_crtx,y_crtx)
elif mode=='pw':
	i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,x_crtx,y_crtx)
else:
	i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode)
	
	
### check if interactions functions look good
if show_intermediate_results:
	y,x = 5,10
	plt.figure()
	ax=plt.subplot(111)
	ax.set_title("Rec. interaction fct")
	im=ax.imshow(i[y*x_LGN+x,:].reshape(y_crtx,x_crtx),interpolation='nearest',cmap='RdBu_r')#,vmin=-1,vmax=1)
	#im=ax.imshow(i,interpolation='nearest',cmap='RdBu_r')
	plt.colorbar(im)
	c = plt.Circle((x,y), radius=0.5, label='patch',color='chartreuse')
	ax.add_patch(c)
	plt.show()
#exit()
	
c_onon = C_onon(x_delta_LGN_LGN,y_delta_LGN_LGN,rc)
c_offoff = C_offoff(x_delta_LGN_LGN,y_delta_LGN_LGN,rc)
c_onoff = C_onoff(x_delta_LGN_LGN,y_delta_LGN_LGN,rc)
a = Arbor(x_delta_LGN_crtx,y_delta_LGN_crtx)
print('ARBOR',np.sum(a>0))
if show_intermediate_results:
	print(c_onon.shape)
	fig=plt.figure(figsize=(10,4))
	ax = fig.add_subplot(121)
	ax.set_title("arbor")
	im=ax.imshow(a.reshape(y_LGN,x_LGN,y_crtx,x_crtx,2)[5,4,:,:,0],interpolation='nearest')
	plt.colorbar(im,ax=ax)
	ax = fig.add_subplot(122)
	ax.set_title("C_onon")
	im2=ax.imshow(c_onon[57,:].reshape(y_crtx,x_crtx),interpolation='nearest')
	plt.colorbar(im2,ax=ax)
	plt.show()
#exit()

## initializing ff weights
np.random.seed(1)
s_init = a*(1 + s_noise*np.random.choice([-1,1],(x_LGN*y_LGN,x_crtx*y_crtx,2)))
s = s_init
s_init *= rhs.synaptic_normalization_factor_init(s_init,a)[None,:,None]
s = s_init
#exit()

## step 5
#s_init = np.ma.array(s_init,mask=None)
#a = np.ma.array(a,mask=None)
s_init,frozen_init = rhs.clip_synapse(s_init,s_max,a,np.zeros_like(s_init,dtype=bool))


## step 1
delta,lambda_normalised = rhs.unconstrained_init(s_init,0,lambda_init,i,\
a,c_onon,c_onoff,c_offoff,target_sigma)
print('lambda_normalised={}'.format(lambda_normalised))


## step 2
s1,st,frozen_ratio,frozen = three_step_method.three_step_method(s_init,\
t0,lambda_normalised,rhs.constrained,frozen_init,i,a,c_onon,c_onoff,c_offoff,s_max)


np.save(conf.data_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}{}.npy'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode,npbc),s1)
print('s1 saved.')
print('frozen ratio at {}.'.format(frozen_ratio[-1]))

if show_intermediate_results:
	SD = np.zeros((DA*y_crtx,DA*x_crtx))
	ensure_path.ensure(conf.movie_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}{}/'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode,npbc))
	for i in range(st.shape[-1]):
		sd = np.pad((st[:,:,0,i]-st[:,:,1,i]).reshape(y_LGN,x_LGN,y_crtx,x_crtx),pad_width=((DA,DA),(DA,DA),(0,0),(0,0)),mode='constant',constant_values=0)
		for icrtx in range(x_crtx):
			for jcrtx in range(y_crtx):
				SD[jcrtx*DA:(jcrtx+1)*DA,icrtx*DA:(icrtx+1)*DA] = sd[jcrtx-6+DA:jcrtx+7+DA,icrtx-6+DA:icrtx+7+DA,jcrtx,icrtx]
		
		fig=plt.figure()
		plt.imshow(SD,interpolation='nearest',cmap='RdBu_r',vmin=-4,vmax=4)
		plt.colorbar()
		fig.savefig(conf.movie_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}{}/{}.png'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode,npbc,i),format='png',dpi=300)#,bbox_inches='tight')
		plt.close()


## visualization of SD = S_on - S_off
sd = (s1[:,:,0]-s1[:,:,1]).reshape(y_LGN,x_LGN,y_crtx,x_crtx)
s1 = s1.reshape(y_LGN,x_LGN,y_crtx,x_crtx,2)
SD = np.zeros((3,DA*y_crtx,DA*x_crtx))
for icrtx in range(x_crtx):
	for jcrtx in range(y_crtx):
		son_ij = np.roll(np.roll(s1[:,:,jcrtx,icrtx,0],shift=y_LGN//2-jcrtx,axis=0),shift=x_LGN//2-icrtx,axis=1)
		sof_ij = np.roll(np.roll(s1[:,:,jcrtx,icrtx,1],shift=y_LGN//2-jcrtx,axis=0),shift=x_LGN//2-icrtx,axis=1)
		SD[0,jcrtx*DA:(jcrtx+1)*DA,icrtx*DA:(icrtx+1)*DA] = \
		 son_ij[y_LGN//2-DA//2:y_LGN//2+DA//2+1, x_LGN//2-DA//2:x_LGN//2+DA//2+1] -\
		 sof_ij[y_LGN//2-DA//2:y_LGN//2+DA//2+1, x_LGN//2-DA//2:x_LGN//2+DA//2+1]
		SD[1,jcrtx*DA:(jcrtx+1)*DA,icrtx*DA:(icrtx+1)*DA] = \
		 son_ij[y_LGN//2-DA//2:y_LGN//2+DA//2+1, x_LGN//2-DA//2:x_LGN//2+DA//2+1]
		SD[2,jcrtx*DA:(jcrtx+1)*DA,icrtx*DA:(icrtx+1)*DA] = \
		 sof_ij[y_LGN//2-DA//2:y_LGN//2+DA//2+1, x_LGN//2-DA//2:x_LGN//2+DA//2+1]
		 
		 
#plt.figure()
#plt.plot(st[:,1,0,:].T,'-')#12

#plt.figure()
#plt.plot(frozen_ratio,'-')#12

#plt.figure()
#plt.imshow(frozen[:,:,1],interpolation='nearest',cmap='RdBu_r')
#plt.colorbar()

#plt.figure()
#plt.imshow(s1[:,:,0]-s1[:,:,1],interpolation='nearest')
#plt.colorbar()

fig = plt.figure(figsize=(18,5))
ax = fig.add_subplot(131)
ax.set_title("S_D")
im = ax.imshow(SD[0,:,:],interpolation='nearest',cmap='binary')
plt.colorbar(im,ax=ax)

ax = fig.add_subplot(132)
ax.set_title("S_on")
im = ax.imshow(SD[1,:,:],interpolation='nearest',cmap='binary')
plt.colorbar(im,ax=ax)

ax = fig.add_subplot(133)
ax.set_title("S_of")
im = ax.imshow(SD[2,:,:],interpolation='nearest',cmap='binary')
plt.colorbar(im,ax=ax)

plt.savefig(conf.image_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}.pdf'.format(x_LGN,\
y_LGN,x_crtx,y_crtx,np.around(rc,2),np.around(rI,2),mode),format="pdf",dpi=300,bbox_inches='tight')
plt.show()






