import numpy as np
# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib.backends.backend_pdf import PdfPages

from tools import image_dir
from tools import tools


def plot_complex_map(complex_map):
	h,w = complex_map.shape
	hsv_map = np.zeros((h,w,3))
	maxmax = np.nanmax(abs(complex_map))
	hsv_map[:,:,0] = ( (np.angle(complex_map)) % (2 * np.pi) ) / np.pi / 2.
	hsv_map[:,:,2] = np.clip(abs(complex_map)/maxmax/0.3,0,1)	# should be >0, and <1
	hsv_map[:,:,1] = 1	# if saturation=1, black is background color
	return mcol.hsv_to_rgb(hsv_map)




def plot_RF(s,**kwargs):
	DA = kwargs["DA"]
	RF,PF = tools.get_RF(s,DA)

	if "plot_size" in kwargs:
		plot_size = kwargs["plot_size"] * DA
		RF = RF[:,:plot_size,:plot_size]
		PF = PF[:,:plot_size,:plot_size]


	cmap_div = "RdBu_r"
	if "cmap_div" in kwargs:
		cmap_div = kwargs["cmap_div"]

	fig = plt.figure(figsize=(18,9))
	## receptive field
	ax = fig.add_subplot(231)
	ax.set_title("S_D")
	im = ax.imshow(RF[0,:,:],interpolation='nearest',cmap=cmap_div)
	plt.colorbar(im,ax=ax)

	ax = fig.add_subplot(232)
	ax.set_title("S_on")
	im = ax.imshow(RF[1,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)

	ax = fig.add_subplot(233)
	ax.set_title("S_of")
	im = ax.imshow(RF[2,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)

	## projective field
	ax = fig.add_subplot(234)
	ax.set_title("S_D (PF)")
	im = ax.imshow(PF[0,:,:],interpolation='nearest',cmap=cmap_div)
	plt.colorbar(im,ax=ax)

	ax = fig.add_subplot(235)
	ax.set_title("S_on (PF)")
	im = ax.imshow(PF[1,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)

	ax = fig.add_subplot(236)
	ax.set_title("S_of (PF)")
	im = ax.imshow(PF[2,:,:],interpolation='nearest',cmap='binary')
	plt.colorbar(im,ax=ax)
	return fig

def plot_RF_grid(RF_list,**kwargs):
	num_panels = len(RF_list)
	ncol = int(np.ceil(np.sqrt(num_panels)))
	nrow = int(np.ceil(1.*num_panels/ncol))

	if "params_list" in kwargs.keys():
		params_list = kwargs["params_list"]
	else:
		params_list = None


	fig = plt.figure(figsize=(ncol*6,nrow*5))
	for i in range(num_panels):
		ax = fig.add_subplot(nrow,ncol,1+i)
		if params_list is not None:
			ax.set_title("rA={:.0f},rI={:.1f},rC={:.1f}".format(*params_list[i]))
		im=ax.imshow(RF_list[i],interpolation='nearest',cmap='RdBu_r')
		plt.colorbar(im,ax=ax)
	return fig



def plot_temporal_behaviour(st,**kwargs):
	DA = kwargs["DA"]
	N = st.shape[1]
	arbor = kwargs["arbor"]
	print("arbor",arbor.shape,N)

	st = st.reshape(-1,N,N,N*N,2)
	arbor = arbor.reshape(N,N,N*N)
	timesteps = st.shape[0]
	t0,t1,t2 = 0,max([3,timesteps//4]),timesteps-1

	ncol,nrow = 6,3
	figt = plt.figure(figsize=(6*ncol,5*nrow))
	ax = figt.add_subplot(nrow,ncol,1)
	ax.plot(st[:,N//2,N//2,::10,0],"-r",rasterized=True)#,label="ON")
	ax.plot(st[:,N//2,N//2,::10,1],"-b",rasterized=True)#,label="OFF")
	# ax.legend(loc="best")
	ax.set_xlabel("Timesteps")
	ax.set_ylabel("FF conn off")

	ax = figt.add_subplot(nrow,ncol,1+ncol)
	for it in [t0,t1,t2]:
		weights_ti = np.sort(st[it,arbor>0,:].flatten())
		ax.plot(weights_ti,np.linspace(0,1,len(weights_ti)),"-",label="t={}".format(it))
	ax.legend(loc="best")
	ax.set_xlabel("FF Weight")
	ax.set_ylabel("Cumulative Distribution")


	st = st.reshape(-1,N,N,N,N,2)
	labels = ["ON","OFF"]

	RF_t0,_ = tools.get_RF(st[t0,...],DA=DA)
	RF_t1,_ = tools.get_RF(st[t1,...],DA=DA)
	RF_t2,_ = tools.get_RF(st[t2,...],DA=DA)
	max_weight = np.nanmax(RF_t2)
	colorbar_max = max_weight

	for i,it in enumerate([t0,t1,t2]):
		opm_ti,_ = tools.get_response(st[it,...,0]-st[it,...,1],DA)
		ax = figt.add_subplot(nrow,ncol,2+i*ncol)
		ax.set_title("T={} OPM".format(it))
		im=ax.imshow(plot_complex_map(opm_ti),interpolation="nearest")
		# im=ax.imshow(np.angle(opm_ti),interpolation="nearest",cmap="hsv")
		plt.colorbar(im,ax=ax,orientation="horizontal")
		ax = figt.add_subplot(nrow,ncol,3+i*ncol)
		ax.set_title("T={} pref ori dist".format(it))
		ax.hist(np.angle(opm_ti,deg=True).flatten()*0.5,bins=20)
		ax.set_xlim(-180,180)

	for i,(it,iRF) in enumerate(zip([t0,t1,t2],[RF_t0,RF_t1,RF_t2])):
		ax = figt.add_subplot(nrow,ncol,4+i*ncol)
		ax.set_title("T={}, ON-OFF".format(it))
		im=ax.imshow(iRF[0,:,:],interpolation="nearest",cmap="RdBu_r",\
					vmin=-colorbar_max,vmax=colorbar_max)
		plt.colorbar(im,ax=ax)
		for j in range(2):
			ax = figt.add_subplot(nrow,ncol,5+j+i*ncol)
			ax.set_title("T={}, {}".format(it,labels[j]))
			im=ax.imshow(iRF[j+1,:,:],interpolation="nearest",cmap="binary",\
						vmin=0,vmax=colorbar_max)
			plt.colorbar(im,ax=ax)
	return figt


def plot_figures(st,**kwargs):
	timesteps = st.shape[0]
	N = kwargs["N"]
	version = kwargs["version"]
	DA = kwargs["DA"]

	st = st.reshape(timesteps,N,N,N,N,2)
	fig_t = plot_temporal_behaviour(st,DA=DA,arbor=kwargs["arbor"])
	pp = PdfPages(image_dir + "output_v{}.pdf".format(version))
	pp.savefig(fig_t,dpi=300,bbox_inches='tight')
	plt.close()


	fig = plot_RF(st[-1,:,:,:,:,:],DA=DA)
	pp.savefig(fig,dpi=300,bbox_inches='tight')
	plt.close()
	pp.close()


