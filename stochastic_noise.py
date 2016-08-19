import numpy as np
import matplotlib.pyplot as plt

Qf = 20 #Grau do polinômio
 #Como nesse teste o Qf não varia é possível manter a variavel "a" fora das iterações
N = np.arange(20,140,5,dtype=np.int)
sigma = np.arange(0,2.05,0.05,dtype=np.float) #?AINDA NÃO SEI 
number_iteration = 10
resultH2 = np.zeros((len(sigma),len(N)))
resultH10 = np.zeros((len(sigma),len(N)))



def iterativeLegendre(n, X):
	if n == 0:
		return 1.0
	elif n == 1:
		return X
	else:
		L = 0
		L0 = np.ones(X.shape)
		L1 = X
		for i in  range(1,n):
			L = ( (2 * i - 1) / i) * X * L1 - ((i - 1) / i) * L0
			L0 = L1
			L1 = L
		return L 

def erro(gn, X, y, a_norm):

	gn = gn + 1
	Eout_n = 0;
	X_n = np.ones((N[i],gn))
	
	for l in range(N[i]):
	
		for m in range(1,gn):
		
			X_n[l,m] = iterativeLegendre(m,X[l])

	w_n = np.linalg.pinv(X_n) * y
				
	a = 0
	b = 0

	for l in range(Qf):
		
		if (l > Qf-1):
			a = 0
		else:
			a = a_norm[l]
		if (l > 3-1):
			b = 0
		else:
			b = w_n[l]
		
		Eout_n = Eout_n + np.sum((np.power((a - b),2) / (2 * l + 1) ))

	return Eout_n	


for iteration in range(number_iteration):
	
	for i in range(len(N)):
	
		for j in range(len(sigma)):
		
			#gerando x no espaço [-1,1]
			X = np.random.uniform(-1, 1, N[i]) #Gerando número aleatório com distribuição uniforme
			
			a = np.random.randn(Qf)			
			e = np.random.randn(N[i])
			y = np.zeros(N[i])
			
			#normalizar coeficientes
			k = 0

			for l in range(Qf):

				k = k + (np.power(a[l],2) / (2 * l + 1))

			a_norm = a / np.sqrt(2*k)
			
			#gerando um y dado um x pertencente ao espaço[-1,1]
			for l in range(N[i]):

				f_value = 0

				for m in range(Qf):

					f_value = f_value + a_norm[m] * iterativeLegendre(m-1,X[l])

				y[l] = f_value + np.sqrt(sigma[j]) * e[l]
			

			resultH2[j,i]  = resultH2[j,i] + erro(2, X, y, a_norm)

			resultH10[j,i] = resultH10[j,i] + erro(10, X, y, a_norm)


resultH2 = np.divide(resultH2, number_iteration)
resultH10 = np.divide(resultH10, number_iteration)

plt.imshow(resultH10- resultH2, cmap="jet", interpolation="gaussian", origin="lower", vmin=-0.2, vmax=0.2, extent=[20,130,0,2], aspect="auto")
plt.colorbar()
plt.show()