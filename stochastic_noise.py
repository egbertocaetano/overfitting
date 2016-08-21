import numpy as np
import matplotlib.pyplot as plt
import math


class Stochastic(object):

	def __init__(self):

		self.Qf = 20 #Grau do polinômio
		self.N = np.arange(20,140,5,dtype=np.int)
		self.sigma = np.arange(0,2.05,0.05,dtype=np.float) 
		self.number_iteration = 20
		self.resultH2 = np.zeros((len(self.sigma),len(self.N)))
		self.resultH10 = np.zeros((len(self.sigma),len(self.N)))

	def iterativeLegendre(self, n, X):
		if n == 0:
			return 1.0
		elif n == 1:
			return X
		else:
			L = 0
			L0 = np.ones(X.shape)
			L1 = X
			for i in  range(2,n+1):
				L = ( (2 * i - 1) / i) * X * L1 - ((i - 1) / i) * L0
				L0 = L1
				L1 = L
			return L 
	@profile		
	def erro(self, gn, X, y, a_norm, step):

		gn = gn + 1
		Eout_n = 0;
		X_n = np.ones((self.N[step],gn))
		
		for l in range(self.N[step]):
		
			for m in range(1,gn):
			
				X_n[l,m] = self.iterativeLegendre(m,X[l])

		w_n = np.dot(np.linalg.pinv(X_n),y)
					
		a = 0
		b = 0

		for l in range(self.Qf+1):
			
			if (l > self.Qf+1):
				a = 0
			else:
				a = a_norm[l]
			if ((l+1) > gn):
				b = 0
			else:
				b = w_n[l]
			
			Eout_n = Eout_n + (math.pow((a - b),2) / ((2 * l) + 1) )

		return Eout_n	
	
	def execute(self):

		for iteration in range(self.number_iteration):
			
			for i in range(len(self.N)):
			
				for j in range(len(self.sigma)):
				
					#gerando x no espaço [-1,1]
					X = np.random.uniform(-1, 1, self.N[i]) #Gerando número aleatório com distribuição uniforme
					
					a = np.random.randn(self.Qf+1)			
					e = np.random.randn(self.N[i])
					y = np.zeros(self.N[i])
					
					#normalizar coeficientes
					k = 0

					for l in range(self.Qf+1):

						k = k + (math.pow(a[l],2) / (2 * l + 1))

					a_norm = np.divide(a,math.sqrt(2*k))
					
					#gerando um y dado um x pertencente ao espaço[-1,1]
					for l in range(self.N[i]):

						f_value = 0

						for m in range(self.Qf+1):

							f_value = f_value + (a_norm[m] * self.iterativeLegendre(m,X[l]))

						y[l] = f_value + (math.sqrt(self.sigma[j]) * e[l])
					

					self.resultH2[j,i]  = self.resultH2[j,i] + self.erro(2, X, y, a_norm, i)

					self.resultH10[j,i] = self.resultH10[j,i] + self.erro(10, X, y, a_norm, i)


		self.resultH2 = np.divide(self.resultH2, self.number_iteration)
		self.resultH10 = np.divide(self.resultH10, self.number_iteration)

		plt.imshow(np.subtract(self.resultH10, self.resultH2), cmap="jet", interpolation="gaussian", origin="lower", vmin=-0.2, vmax=0.2, extent=[20,130,0,2], aspect="auto")
		plt.colorbar()
		plt.show()

stochastic = Stochastic()
stochastic.execute()	