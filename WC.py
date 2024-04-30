# In the name of God
# Wilson and Cowan model in nodes of a WS network without delay.
# "Sheida_Kazemi@modares.ac.ir"

from smallworld.draw import draw_network
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

# Generate a WS network:
k = 80  # The number of nodes.
watts_strogatz = nx.watts_strogatz_graph(k, 20, 0.2)
draw_network(watts_strogatz, 20)
M = nx.to_numpy_matrix(watts_strogatz)  # Adjacency matrix
plt.close()
rewire = 0.2

power_coeff = 1
alpha = 0.325 ** power_coeff  # coupling strength
print(alpha)
rp = 18  # The number of repetition (changes from 1 to 10)

# Functions:
sig = lambda x: 1/(1+np.exp(-x))

f0 = 0


# Parameters:
CEE = 8
CEI = 16
CIE = 8
CII = 4

thrE = 2
thrI = 8
betaE = .8
betaI = .8

TouE = 0.125
TouI = 0.25
Tou = TouI/TouE

sigma = 1  # (***** initial noise for creating completely uncorrelated initial dynamics ***)
sigmaF = 1  # = sqrt(2D)   (*** the stochastic term of noise = sqrt(2D) *** )

dt = 0.0001  # Time step

T1 = 200  # Running time
Tfix1 = min(5., T1)  # Time  for equilibrium

T = range(int(T1 / dt))
t = np.linspace(0, T1, int(T1 / dt) + 1)
print((len(t)))

Tfix = int(Tfix1 / dt)  # Time Steps for equilibrium
TS = t[Tfix:]

rparray_m = np.zeros(rp)
rparray_v = np.zeros(rp)

E = np.zeros((k, len(t)))
I = np.zeros((k, len(t)))


# initial conditions:
E[:, 0] = np.random.uniform(0, 1, k)
I[:, 0] = np.random.uniform(0, 1, k)

P = np.zeros(k)
Q = np.zeros(k)
noise = np.zeros((k, len(t)))

startTime = time.time()

for i in range(k):
    noise[i, :] = np.random.normal(0, sigma * np.sqrt(dt), len(t))

for tt in T:
    #rd = np.random.randint(300, 600)
    #for ii in range(k):
        #P[ii] = rd / 1000
    #rd = 300 * np.random.rand()
    #P[j] = rd / 1000
    if tt*dt >= 5/2.:
        sigma = sigmaF
    if (tt + 1) % 1000 == 0:
        print(tt, ' steps of ', len(T))
        ti = time.time() - startTime
        print('Elapsed time = \t', ti, ' sec\nTotal time = \t', ti / tt * len(T), 'sec\n')

    sm = np.matmul(M, E[:, tt])
    E[:, tt + 1] = (dt * (1/TouE) * (-E[:, tt] + sig(betaE * (CEE * E[:, tt] - CIE * I[:, tt] - thrE + f0 + noise[:, tt] + alpha *
                             sm[:])))) + E[:, tt]
    I[:, tt + 1] = (dt * (1/TouI) * (-I[:, tt] + sig(betaI * (CEI * E[:, tt] - CII * I[:, tt] - thrI
                                                                  )))) + I[:, tt]

lag = 10
EOut = []
IOut = []
EOutM = []
IOutM = []
#yOutM = []
tOut = []
ii = -1
for i in range(- Tfix + len(T)):
    ii += 1
    if ii % lag == 0:
        tOut.append(i * dt)
        EOut.append(E[:, i + Tfix])
        IOut.append(I[:, i + Tfix])
        EOutM.append(np.mean(E[:, i + Tfix]))
        IOutM.append(np.mean(I[:, i + Tfix]))
EOut = np.array(EOut)
IOut = np.array(IOut)
EOutM = np.array(EOutM)
IOutM = np.array((IOutM))
EOut = np.transpose(EOut)
IOut = np.transpose(IOut)
EOutM = np.transpose(EOutM)
IOutM = np.transpose(IOutM)
print("Shape_E", np.shape(EOut))
print("Shape_t", np.shape(tOut))

np.save('wc_output_alpha' + str(alpha) + '_rp' + str(rp) + '_rewire' + str(rewire) + '_Regular1_paper3_sigma' + str(sigma), EOut)
#np.save('wc_mean_alpha' + str(alpha) + '_rp' + str(rp), np.mean(E[:, Tfix:], axis=0))

plt.figure()
#for j, color in enumerate(['r', 'k', 'g']):
    #sn = int((np.random.random(1)) * k)
    #plt.plot(tOut, EOut[sn, Tfix:])
plt.plot(tOut, EOut[70, :], color='red')
plt.plot(tOut, IOut[70, :], color='blue')
plt.ylabel('values')
plt.xlabel('time')
plt.title("The output of WC model")

plt.figure()
plt.plot(EOutM, IOutM)
plt.xlabel("E")
plt.ylabel("I")
plt.title("Phase_space")


plt.show()