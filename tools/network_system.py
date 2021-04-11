import numpy as np
from scipy import linalg
from copy import copy


def distance(delta,N):
	''' assume periodic boundary conditions'''
	signs = np.sign(delta)
	# return np.where(np.abs(delta) > 0.5, -signs*(1 - np.abs(delta)), delta)
	return np.where(np.abs(delta) > (0.5 * N),\
	 		-signs*(N - np.abs(delta)), delta)

def gaussian(x,y,s):
	return np.exp(-(x**2+y**2)/2./s**2)#1./2.*np.pi/s**2*


class Network:
	def __init__(self, from_size, to_size, Version):
		self.from_size = from_size
		self.to_size = to_size		
		self.rng = np.random.RandomState(20210311 + Version)

	def create_matrix(self, conn_params, profile, **kwargs):
		grid = np.linspace(0,self.from_size[0],self.from_size[0],endpoint=False)
		xfrom,yfrom = np.meshgrid(grid,grid)
		grid = np.linspace(0,self.to_size[0],self.to_size[0],endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		xdelta = distance(xto[None,None,:,:]-xfrom[:,:,None,None],self.to_size[0])
		ydelta = distance(yto[None,None,:,:]-yfrom[:,:,None,None],self.to_size[0])
		
		if profile in ("Gaussian",):
			sigma = conn_params["sigma"]
			ampl = conn_params["ampl"]
			disc_gaussian = gaussian(xdelta,ydelta,sigma) #/ np.prod(self.from_size)
			norm_factor = np.sum(disc_gaussian,axis=(2,3))[:,:,None,None]
			conn_matrix = ampl * disc_gaussian #/ norm_factor
		elif profile=="Gabor":
			sigma = conn_params["sigma"] * np.nanmax(xdelta)
			ampl = conn_params["ampl"]
			theta = conn_params["theta"]
			psi = conn_params["psi"]
			disc_gaussian = ampl * gaussian(xdelta,ydelta,sigma)
			Lambda = 2*np.pi/conn_params["freq"] * np.nanmax(xdelta)
			gamma = 1.
			sigma_x = sigma
			sigma_y = float(sigma) / gamma

			# Bounding box
			nstds = 3  # Number of standard deviation sigma
			xmax = np.max(np.concatenate([np.abs(nstds * sigma_x * np.cos(theta)),\
									   np.abs(nstds * sigma_y * np.sin(theta))]))
			xmax = np.ceil(max(1, xmax))
			ymax = np.max(np.concatenate([np.abs(nstds * sigma_x * np.sin(theta)),\
					 				   np.abs(nstds * sigma_y * np.cos(theta))]))
			ymax = np.ceil(max(1, ymax))
			print("xd",np.nanmax(xdelta),xmax,ymax,sigma,Lambda)
			x = xdelta * 2 * xmax / np.nanmax(xdelta)
			y = ydelta * 2 * ymax / np.nanmax(ydelta)
			# Rotation
			x_theta = x * np.cos(theta[None,None,:,:]) - y * np.sin(theta[None,None,:,:])
			y_theta = x * np.sin(theta[None,None,:,:]) + y * np.cos(theta[None,None,:,:])
			if isinstance(psi,np.ndarray):
				conn_matrix =\
					 np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) *\
				 	 np.cos(2 * np.pi / Lambda * x_theta + psi[:,:,None,None])
			else:
				conn_matrix =\
					 np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) *\
				 	 np.cos(2 * np.pi / Lambda * x_theta + psi)
		else:
			print("Specified connectivity profile ({}) not found.".format(profile))

		self.conn_matrix = conn_matrix.reshape(np.prod(self.from_size),\
							np.prod(self.to_size))
		
		if "r_A" in kwargs:
			profile_A = kwargs.get("profile_A", "heaviside")
			arbor = self.create_arbor(radius=kwargs["r_A"],profile=profile_A)
			self.conn_matrix[np.logical_not(arbor)] = 0.
		
		return self.conn_matrix


	def create_arbor(self, radius, profile="heaviside"):
		"""
		arbor function: crude model of overlap of geniculocortical terminal
		arbors with cortical dendritic arbors
		"""
		grid = np.linspace(0,self.from_size[0],self.from_size[0],endpoint=False)
		xfrom,yfrom = np.meshgrid(grid,grid)
		grid = np.linspace(0,self.to_size[0],self.to_size[0],endpoint=False)
		xto,yto = np.meshgrid(grid,grid)
		xdelta = distance(xto[None,None,:,:]-xfrom[:,:,None,None],self.to_size[0])
		ydelta = distance(yto[None,None,:,:]-yfrom[:,:,None,None],self.to_size[0])

		d = np.sqrt(xdelta**2 + ydelta**2)		## absolute distance |vec(x)-vec(y)|
		d = d.reshape(np.prod(self.to_size),np.prod(self.from_size))
		if profile=="heaviside":
			arbor = (np.around(d,2)<=radius).astype(float)

		elif profile=="gaussian":
			arbor = gaussian(d,0,radius)
			arbor[np.around(d,2)>radius] = 0.0
		
		elif profile=="overlap":
			d_unique = np.unique(d[np.around(d,2)<=radius])
			cA = 0.5
			N = radius*(cA+2)

			delta_bin = 0.05
			i,j = np.meshgrid(np.arange(-(self.to_size[0]//2),self.to_size[0]//2+delta_bin,\
				delta_bin),\
				np.arange(-(self.from_size[0]//2),self.from_size[0]//2+delta_bin,delta_bin))
			delta1 = np.sqrt(i**2+j**2)
			delta2 = np.sqrt((d_unique[:,None,None]-i[None,:,:])**2+(j[None,:,:])**2)
			circles = (delta1 < radius)[None,:,:]*(delta2 < (radius*cA))
			norm = np.sum(delta2 < (radius*cA),axis=(1,2))
			overlap = 1.*np.sum(circles,axis=(1,2))/norm

			arbor = np.zeros_like(d,dtype=float)
			for iduni in range(d_unique.size):
				arbor[d==d_unique[iduni]] = overlap[iduni]

		elif profile=="linear_falloff":
			arbor = linear_arbor_falloff(np.around(d,2),radius=radius,\
										 inner_radius=radius*0.5)
			arbor[np.around(d,2)>radius] = 0.0

		else:
			print("Specified arbor profile ({}) not found.".format(profile))

		return arbor
