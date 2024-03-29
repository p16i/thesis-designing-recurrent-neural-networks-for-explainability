# this is draw from http://heatmapping.org/tutorial/
import numpy

# -------------------------
# Feed-forward network
# -------------------------
class Network:

	def __init__(self,layers):
		self.layers = layers

	def forward(self,Z):
		for l in self.layers:
			Z = l.forward(Z)
		return Z

	def gradprop(self,DZ):
		for l in self.layers[::-1]:
			DZ = l.gradprop(DZ)
		return DZ

# -------------------------
# ReLU activation layer
# -------------------------
class ReLU:

	def forward(self,X):
		self.Z = X>0
		return X*self.Z

	def gradprop(self,DY):
		return DY*self.Z

# -------------------------
# Fully-connected layer
# -------------------------
class Linear:

	def __init__(self,name):
		self.W = numpy.loadtxt(name+'-W.txt')
		self.B = numpy.loadtxt(name+'-B.txt')

	def forward(self,X):
		self.X = X
		return numpy.dot(self.X,self.W)+self.B

	def gradprop(self,DY):
		self.DY = DY
		return numpy.dot(self.DY,self.W.T)

# -------------------------
# Sum-pooling layer
# -------------------------
class Pooling:

	def forward(self,X):
		self.X = X
		self.Y = 0.5*(X[:,::2,::2,:]+X[:,::2,1::2,:]+X[:,1::2,::2,:]+X[:,1::2,1::2,:])
		return self.Y

	def gradprop(self,DY):
		self.DY = DY
		DX = self.X*0
		for i,j in [(0,0),(0,1),(1,0),(1,1)]: DX[:,i::2,j::2,:] += DY*0.5
		return DX

# -------------------------
# Convolution layer
# -------------------------
class Convolution:

    def __init__(self,name):
        wshape = map(int,list(name.split("-")[-1].split("x")))
        wshape = list(wshape)
        self.W = numpy.loadtxt(name+'-W.txt').reshape(wshape)
        self.B = numpy.loadtxt(name+'-B.txt')

    def forward(self, X):

        self.X = X
        mb,wx,hx,nx = X.shape # minibatch, input width, input height, no. output channel
        ww,hw,nx,ny = self.W.shape # filter width, filter height, no.input channel, no.output channel
        wy,hy = wx-ww+1,hx-hw+1 # no padding here

        Y = numpy.zeros([mb,wy,hy,ny],dtype='float32') # minibatch, output width, output height, no.output channel

        for i in range(ww): # filter width : 0,1,2,3,4
            for j in range(hw): #filter height
                Y += numpy.dot(X[:,i:i+wy,j:j+hy,:],self.W[i,j,:,:])
				#               mb,0:28, 0:28, o.ch,      0,0, i.ch, oy.ch
		        #               mb,0:28, 1:29, o.ch,      0,1, i.ch, oy.ch

        return Y+self.B

    def gradprop(self,DY):

        self.DY = DY
        mb,wy,hy,ny = DY.shape
        ww,hw,nx,ny = self.W.shape

        DX = self.X*0

        for i in range(ww):
            for j in range(hw):
                DX[:,i:i+wy,j:j+hy,:] += numpy.dot(DY,self.W[i,j,:,:].T)

        return DX

