# In the name of God
# Jansen and Rit model in nodes of a WS network without delay.
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

alpha = 3  # coupling strength
rp = 10  # The number of repetition (changes from 1 to 10)

# Define network parameters

# Parameters:

e0 = 2.5  # s-1
r = 0.56  # slop 0.56 /mV
v0 = 6  # amplitude mV


A = 3.25  # mv [2,14]
B = 22  # mv [10,30]
a = 100  # s-1
b = 50  # s-1
C = 135  # 12.93 / 2 / e0 * a / A /r
C1 = 1.0 * C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C

a_noise = 120
b_noise = 120
p = 0.5 * (a_noise + b_noise)
f0 = p  # 150  (**** the mean of input signal which is added to the equation as deterministic input****)
sigma = 1  # (***** initial noise for creating completely uncorrelated initial dynamics ***)
sigmaF = 1  # = sqrt(2D)   (*** the stochastic term of noise = sqrt(2D) *** )


sig = lambda v: (2 * e0) / (1 + np.exp(r * (v0 - v)))

dt = 0.0001  # Time step
#sqdt = np.sqrt(dt)

T1 = 200  # Running time
Tfix1 = min(5., T1)  # Time  for equilibrium

T = range(int(T1 / dt))
t = np.linspace(0, T1, int(T1 / dt) + 1)

Tfix = int(Tfix1 / dt)  # Time Steps for equilibrium
TS = t[Tfix:]

y0 = np.zeros((k, len(t)))
y1 = np.zeros((k, len(t)))
y2 = np.zeros((k, len(t)))
y3 = np.zeros((k, len(t)))
y4 = np.zeros((k, len(t)))
y5 = np.zeros((k, len(t)))
yy = np.zeros((k, len(t)))

# initial conditions:
y0[:, 0] = np.random.uniform(5, 10, k)
y1[:, 0] = np.random.uniform(5, 10, k)
y2[:, 0] = np.random.uniform(5, 10, k)
y3[:, 0] = np.random.uniform(5, 10, k)
y4[:, 0] = np.random.uniform(5, 10, k)
y5[:, 0] = np.random.uniform(5, 10, k)
yy[:, 0] = y1[:, 0] - y2[:, 0]  # The output of model


noise = np.zeros((k, len(t)))

startTime = time.time()
noise[:, 0] = 0

for i in range(k):
    noise[i, :] = np.random.normal(0, sigma * np.sqrt(dt), len(t))

for tt in T:
    # p = brownian_noise(k)
    if tt*dt >= 5/2.:
        sigma = sigmaF
    if (tt + 1) % 1000 == 0:
        print(tt, ' steps of ', len(T))
        ti = time.time() - startTime
        print('Elapsed time = \t', ti, ' sec\nTotal time = \t', ti / tt * len(T), 'sec\n')


    sigyy = sig(y1[:, tt] - y2[:, tt] + y4[:, tt] * dt - y5[:, tt] * dt)
    sm = np.matmul(M, sigyy)


    y0[:, tt + 1] = y0[:, tt] + y3[:, tt] * dt + 0.5 * (A * a * sig(y1[:, tt] - y2[:, tt]) - 2 * a * y3[:, tt] - (a ** 2) * y0[:, tt]) * (dt**2)

    y1[:, tt + 1] = y1[:, tt] + y4[:, tt] * dt + 0.5 * (A * a * (f0 + C2 * sig(C1 * y0[:, tt]) + alpha * sm[:]) - 2 * a * y4[:, tt] - (a ** 2) * y1[:, tt]) * (dt**2)

    y2[:, tt + 1] = y2[:, tt] + y5[:, tt] * dt + 0.5 * (B * b * C4 * sig(C3 * y0[:, tt]) - 2 * b * y5[:, tt] - (b ** 2) * y2[:, tt]) * (dt**2)

    y3[:, tt + 1] = A * a * sig(y1[:, tt] - y2[:, tt] + y4[:, tt] * dt - y5[:, tt] * dt) - 2 * a * y3[:, tt] - (a ** 2) * (y0[:, tt] + y3[:, tt] * dt)
    y3[:, tt + 1] = y3[:, tt] + 0.5 * (A * a * sig(y1[:, tt] - y2[:, tt]) - 2 * a * y3[:, tt] - (a ** 2) * y0[:, tt]) * (1-2*a*dt) * dt + 0.5 * y3[:, tt + 1] * dt


    y4[:, tt + 1] = A * a * (f0 + C2 * sig(C1 * y0[:, tt] + C1 * y3[:, tt] * dt) + alpha * sm[:]) - 2 * a * (y4[:, tt]) - (a ** 2) * (y1[:, tt] + y4[:, tt] * dt)
    y4[:, tt + 1] = y4[:, tt] + 0.5 * (A * a * (f0 + C2 * sig(C1 * y0[:, tt]) + alpha * sm[:]) - 2 * a * y4[:, tt] - (a ** 2) * y1[:, tt]) * (1-2*a*dt) * dt + 0.5 * y4[:, tt + 1] * dt + A * a * (1-a*dt) * noise[:, tt]

    y5[:, tt + 1] = B * b * (C4 * sig(C3 * y0[:, tt] + C3 * y3[:, tt] * dt)) - 2 * b * y5[:, tt] - (b ** 2) * (y2[:, tt] + y5[:, tt] * dt)
    y5[:, tt + 1] = y5[:, tt] + 0.5 * (B * b * C4 * sig(C3 * y0[:, tt]) - 2 * b * y5[:, tt] - (b ** 2) * y2[:, tt]) * (1-2*b*dt) * dt + 0.5 * y5[:, tt + 1] * dt

    yy[:, tt + 1] = y1[:, tt + 1] - y2[:, tt + 1]

print('writing output ...')

lag = 10
yOut = []
yOutM = []
tOut = []
ii = -1
for i in range(- Tfix + len(T)):
    ii += 1
    if ii % lag == 0:
        tOut.append(i * dt)
        yOut.append(yy[:, i + Tfix])
        yOutM.append(np.mean(yy[:, i + Tfix]))
yOut = np.array(yOut)
yOutM = np.array(yOutM)
yOut = np.transpose(yOut)
yOutM = np.transpose(yOutM)

np.save('jr_output_alpha' + str(alpha) + '_rp' + str(rp) + '_rewire' + str(rewire) + '_Regular1_sigma' + str(sigma), yOut)  # yy[:, Tfix:])

print('writing output finished.')


plt.figure()  # plot the mean of output in time
#plt.plot(TS, np.mean(yy[:, Tfix:], axis=0))
plt.plot(tOut, yOutM)
plt.xlabel('time')
plt.ylabel('values')
plt.title("The mean of rate")

plt.figure()  # plot phase diagram
#plt.plot(yy[-1, (Tfix + int(1/dt)):int(T1 * dt/2)], y4[-1, (Tfix + int(1/dt)):int(T1 * dt/2)]-y5[-1, (Tfix + int(1/dt)):int(T1 * dt/2)])
plt.plot(np.mean(yy[:, Tfix + int(1/dt):], axis=0), np.mean(y4[:, Tfix + int(1/dt):]-y5[:, Tfix + int(1/dt):], axis=0), linewidth=0.4, alpha=0.5)
plt.xlabel('y1-y2 (mv)')
plt.ylabel('y4-y5 (mv/s)')
#plt.title("Phase Diagram")
plt.title('(' + r'$ \alpha= $' + str(alpha) + ')', fontsize=15)
plt.savefig("jr_fig_phase.d_alpha" + str(alpha) + '_rp' + str(rp) + '_rewire' + str(rewire) + '_Regular1' + '.png', dpi=300)

plt.show()

