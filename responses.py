import numpy as np
import bettina.modeling.miller94.conf as conf
from bettina.modeling.miller94.functions import I,distance

def grating(x,y,k1,k2,phi):
	return np.sin(2*np.pi*(k1*x+k2*y)/64+phi)

def corr(x,y):
	xnorm = (x - x.mean()/x.std()
	ynorm = (y - y.mean())/y.std()
	return np.dot(xnorm.flatten(),y.flatten())/x.size

def get_response_mov(rc,rI,mode):
	x_LGN,y_LGN = conf.x_LGN,conf.y_LGN
	x_crtx,y_crtx = conf.x_cortex,conf.y_cortex

	s1 = np.load(conf.data_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}.npy'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode))
	nframes = s1.shape[3]
	sd = (s1[:,:,0,:] - s1[:,:,1,:]).reshape(y_LGN,x_LGN,y_crtx,x_crtx,nframes)


	factor = 2
	delta_bins = 10
	sdfft = np.abs(np.fft.fftshift(np.fft.fftn(sd,s=(factor*y_LGN,factor*x_LGN),axes=(0,1)),axes=(0,1)))
	sdfft = sdfft[:y_LGN*factor//2,...]
	sdfft_long = sdfft.reshape(factor*factor*x_LGN*y_LGN//2,y_crtx,x_crtx,nframes)

	kx,ky = np.meshgrid(np.arange(-factor*x_LGN/2.,factor*x_LGN/2.),np.arange(0,factor*y_LGN/2.)[::-1])
	angles = np.arctan2(ky,kx)*180/np.pi + (np.arctan2(ky,kx)<0)*360
	ori_bins = (np.searchsorted(np.arange(0,180,delta_bins),angles,side='right')).reshape(factor*factor*x_LGN*y_LGN//2)

	Rn = np.empty((180//delta_bins,y_crtx,x_crtx,nframes))*np.nan
	for ibin in range(1,1+180//delta_bins):
		Rn[ibin-1,...] = np.max(sdfft_long[ori_bins==ibin,...],axis=0)

	opm = np.sum(np.exp(1j*np.linspace(0,2*np.pi,num=180//delta_bins,endpoint=False))[:,None,None,None]*Rn,axis=0)
	rms = np.sum(Rn,axis=0)
	opm = opm/rms
	return opm,Rn

def get_response(rc,rI,mode):
	x_LGN,y_LGN = conf.x_LGN,conf.y_LGN
	x_crtx,y_crtx = conf.x_cortex,conf.y_cortex

	s1 = np.load(conf.data_dir+'SD/{}{}x{}{}_rc{}_rI{}_{}.npy'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode))
	sd = (s1[:,:,0] - s1[:,:,1]).reshape(y_LGN,x_LGN,y_crtx,x_crtx)

	factor = 2
	delta_bins = 10
	sdfft = np.abs(np.fft.fftshift(np.fft.fftn(sd,s=(factor*y_LGN,factor*x_LGN),axes=(0,1)),axes=(0,1)))
	sdfft = sdfft[:y_LGN*factor//2,:]
	sdfft_long = sdfft.reshape(factor*factor*x_LGN*y_LGN//2,y_crtx,x_crtx)

	kx,ky = np.meshgrid(np.arange(-factor*x_LGN/2.,factor*x_LGN/2.),np.arange(0,factor*y_LGN/2.)[::-1])
	angles = np.arctan2(ky,kx)*180/np.pi + (np.arctan2(ky,kx)<0)*360
	ori_bins = (np.searchsorted(np.arange(0,180,delta_bins),angles,side='right')).reshape(factor*factor*x_LGN*y_LGN//2)

	Rn = np.empty((180//delta_bins,y_crtx,x_crtx))*np.nan
	for ibin in range(1,1+180//delta_bins):
		Rn[ibin-1,:,:] = np.max(sdfft_long[ori_bins==ibin,:,:],axis=0)

	opm = np.sum(np.exp(1j*np.linspace(0,2*np.pi,num=180//delta_bins,endpoint=False))[:,None,None]*Rn,axis=0)
	rms = np.sum(Rn,axis=0)
	opm = opm/rms
	return opm,Rn

if __name__ == "__main__":
	#import matplotlib
	#matplotlib.use('agg')
	import bettina.tools.polarplotter as pp
	import matplotlib.pyplot as plt
	import matplotlib.cm as cm
	from bettina.modeling.miller94.conf import params
	
	DA = params['DA']
	x_LGN,y_LGN = conf.x_LGN,conf.y_LGN
	x_crtx,y_crtx = conf.x_cortex,conf.y_cortex

	x,y = np.meshgrid(np.arange(0,x_crtx,1),np.arange(0,y_crtx,1))
	coord_crtx = np.dstack([x,y]).reshape(x_crtx*y_crtx,2)
	x_delta_crtx_crtx = distance(coord_crtx[:,0,None], coord_crtx[None,:,0], x_crtx)
	y_delta_crtx_crtx = distance(coord_crtx[:,1,None], coord_crtx[None,:,1], y_crtx)
	##PARAMETERS TO VARY
	rI = 0.3
	rc = 0.268
	mode = 'pw'

	opm,Rn = get_response(rc,rI,mode)

	if mode=='data':
		ferret = 1581
		date = '2013-12-18'
		tseries = 1
		VERSION = 1
		i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,ferret,date,tseries,VERSION,x_crtx,y_crtx)
	elif mode=='Rgauss':
		i,oris,sel = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,x_crtx,y_crtx)
	elif mode=='Rgauss2':
		i,oris,sel = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,x_crtx,y_crtx)
	elif mode=='pw':
		i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode,x_crtx,y_crtx)
	else:
		i = I(x_delta_crtx_crtx,y_delta_crtx_crtx,rI,mode)

	np.corrcoef(i[y*x_LGN+x,:],)

## VISUALIZATION
	img, cbar, custom_map = pp.colorize(opm)
	plt.figure(figsize=(14,9))
	plt.subplot(241)
	plt.title('opm (result)')
	plt.imshow(img,interpolation='nearest', cmap=custom_map)
	plt.subplot(242)
	plt.title('opm shifted')
	test = np.angle(opm)*0.5+np.pi*(np.angle(opm)<0)
	test2 = (np.angle(opm)%(2*np.pi))/2
	plt.imshow(test,interpolation='nearest', cmap=custom_map,vmin=0,vmax=np.pi)
	if mode in ('Rgauss','Rgauss2'):
		img0, cbar, custom_map = pp.colorize(sel*np.exp(2*1j*oris))
		plt.subplot(243)
		plt.title('random opm')
		plt.imshow(img0,interpolation='nearest', cmap='binary')
	plt.subplot(244)
	plt.title('colorbar')
	plt.imshow(cbar, extent = [0, 180, 0, 1], aspect=180.)
	plt.xlabel('Phase [deg]')
	plt.ylabel('Abs Value')
	
	plt.subplot(245)
	plt.title('cardinal')
	plt.imshow(np.real(opm),interpolation='nearest',cmap='binary')
	plt.subplot(246)
	plt.title('oblique')
	plt.imshow(np.imag(opm),interpolation='nearest',cmap='binary')
	plt.subplot(247)
	plt.title('spectrum(opm)')
	plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(opm))),interpolation='nearest',cmap='binary')
	ax=plt.subplot(248)
	plt.title('Example of I(x,y)')
	y,x = 4,10
	ax.imshow(i[y*x_LGN+x,:].reshape(y_crtx,x_crtx),interpolation='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
	c = plt.Circle((x,y), radius=0.5, label='patch',color='chartreuse')
	ax.add_patch(c)
	plt.savefig(conf.image_dir+'resp/{}{}x{}{}_rc{}_rI{}_{}_{}.pdf'.format(x_LGN,y_LGN,x_crtx,y_crtx,rc,rI,mode,DA),format="pdf",dpi=300,bbox_inches='tight')
	plt.show()

