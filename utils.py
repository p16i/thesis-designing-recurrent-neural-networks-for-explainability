# This is drawn from http://heatmapping.org/tutorial/

import numpy,PIL,PIL.Image

lowest = -1.0
highest = 1.0

# --------------------------------------
# Sampling data
# --------------------------------------

def getMNISTsample(N=12,seed=None,path=''):

	fx = '%s/t10k-images-idx3-ubyte'%path
	ft = '%s/t10k-labels-idx1-ubyte'%path

	X  = numpy.fromfile(open(fx),dtype='ubyte',count=16+784*10000)[16:].reshape([10000,784])
	T  = numpy.fromfile(open(ft),dtype='ubyte',count=8+10000)[8:]
	T  = (T[:,numpy.newaxis]  == numpy.arange(10))*1.0

	if seed==None: seed=numpy.random
	else: seed=numpy.random.mtrand.RandomState(seed)

	R = seed.randint(0,len(X),[N])
	X,T = X[R],T[R]

	return X/255.0*(highest-lowest)+lowest,T

# --------------------------------------
# Color maps ([-1,1] -> [0,1]^3)
# --------------------------------------

def heatmap(x):

	x = x[...,numpy.newaxis]

	r = 0.9 - numpy.clip(x-0.3,0,0.7)/0.7*0.5
	g = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4
	b = 0.9 - numpy.clip(x-0.0,0,0.3)/0.3*0.5 - numpy.clip(x-0.3,0,0.7)/0.7*0.4

	return numpy.concatenate([r,g,b],axis=-1)

def graymap(x):

	x = x[...,numpy.newaxis]
	return numpy.concatenate([x,x,x],axis=-1)*0.5+0.5

# --------------------------------------
# Visualizing data
# --------------------------------------

def visualize(x,colormap,name):

	N = len(x); assert(N<=16)

	x = colormap(x/numpy.abs(x).max())

	# Create a mosaic and upsample
	x = x.reshape([1,N,28,28,3])
	x = numpy.pad(x,((0,0),(0,0),(2,2),(2,2),(0,0)),'constant',constant_values=1)
	x = x.transpose([0,2,1,3,4]).reshape([1*32,N*32,3])
	x = numpy.kron(x,numpy.ones([2,2,1]))

	PIL.Image.fromarray((x*255).astype('byte'),'RGB').save(name)


