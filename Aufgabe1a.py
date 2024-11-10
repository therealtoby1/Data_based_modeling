from scipy.io import loadmat
from scipy.signal import freqz, dlti, dlsim
import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
from Functionfile import ARX
#---------------------------------------------------------------------------------------------------------------------------
#extracting the Data from the .mat file file
data=loadmat("C:\\KIT\\Master_KIT\\DMC\\20240613_hoerter\\20240613_hoerter\\ex1_data.mat")

#show what the values are:
print(data.keys())
u1=np.array(data['u'].T)
y1=np.array(data['y'])
t1=np.array(data['t'].T)
para=(data['para'])




na=3
nb=2
Ts=np.max(t1)/(len(t1)-1)
#print(Ts)
#---------------------------------------------------------------------------------------------------------------------------

#initial Values for the estimation
Pini=np.array(10**7*np.eye(na+nb+1))
pini=np.zeros(na+nb+1)

sky=np.zeros(na)
sku=np.zeros(nb+1)
skT = np.hstack((sky.T, sku.T))


#---------------------------------------------------------------------------------------------------------------------------
#recursive algorithm

for k in range(0,len(t1)):
    pini,Pini,sku,sky,skT =ARX(pini,Pini,sku,sky,y1,u1,skT,na,nb,k)
    
#---------------------------------------------------------------------------------------------------------------------------
#Estimation and discrete transfer function

b = pini[na:].flatten()
a = np.concatenate(([1], -pini[:na].flatten()))
system=dlti(b,a,dt=Ts)
#print(system,pini)
#---------------------------------------------------------------------------------------------------------------------------
#simulating system response
t = np.arange(len(t1)) * Ts
tout, yest = dlsim(system, u1,t)  #simulation for dataset 1

#tout2, yest2= dlsim(system,u2,t)                       #simulation for dataset2


#---------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(t, u1, label='Input Signal training data')
plt.title('Input Signal')
plt.xlabel('Time')
plt.ylabel('Input')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 8))

# Plot the estimated output response
plt.subplot(3, 1, 1)  # 2 rows, 1 column, subplot 1
plt.plot(t, yest, label='Estimated Output ')
plt.title('Estimated vs. Real Output training data')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid()

# Plot the real output
plt.subplot(3, 1, 2)  # 2 rows, 1 column, subplot 2
plt.plot(t, y1, label='Real Output')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)  # 2 rows, 1 column, subplot 3
plt.plot(t, y1-yest, label='estimation error')
plt.xlabel('Time')
plt.ylabel('Output error')
plt.legend()
plt.grid()
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()


