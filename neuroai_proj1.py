import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------
# Part A Bio neuron
#----------------------------------
tau = 10 # shows how fast/slow neuron change happens (10 ms)
v_rest = -65
v_reset = -70
v_thresh = -50
dt = 1
time = np.arange(0, 200, dt)
I = 10 #how much water is being poured in (input current)

v = np.ones(len(time)) * v_rest
spikes = []

for t in range(1, len(time)):
    dv = (-(v[t-1] - v_rest) + I) / tau 
    v[t] = v[t-1] + dv * dt #take previous voltage add smallest chnage times by the time step

if v[t] >= v_thresh:
    v[t] = v_reset
    spikes.append(time[t]) #append =add smt at the end of LIST



plt.figure()
plt.plot(time, v)
plt.title("Leaky Integrate-and-Fire (LIF) Neuron")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.show()

print(f"LIF neuron fired {len(spikes)} times in {time[-1]} ms")


#---------------------------------------------------
# Part B: Artificial neuron (Perceptron)

np.random.seed(0)
X = np.vstack([np.random.randn(50,2) + np.array([1,1]), 
              np.random.randn(50,2) + np.array([-1,-1])])
Y = np.hstack((np.ones(50), -np.ones(50)))


w = np.zeros(2)
b = 0
lr = 0.1

for epoch in range (20):
    for i in range(len(Y)):
        if Y[i]*(np.dot(X[i], w)+b) <= 0:
            w += lr * Y[i] * X[i]
            b += lr * Y[i]


plt.figure()
plt.scatter(X[:,0], X[:,1], c=Y, cmap= 'bwr')
xline = np.linspace(-3, 3, 100)
plt.plot(xline, -(w[0]*xline +b)/w[1], 'k--')
plt.title("Perceptron Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()