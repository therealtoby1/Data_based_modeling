from scipy.io import loadmat
from scipy.signal import freqz, dlti, dlsim
import numpy as np
import matplotlib.pyplot as plt
from Functionfile import Hammerstein,solveHammerstein
#---------------------------------------------------------------------------------------------------------------------------


#extracting the Data from the .mat file file
data=loadmat("C:\\KIT\\Master_KIT\\DMC\\20240613_hoerter\\20240613_hoerter\\ex1_data.mat")

#show what the values are:
print(data.keys())
u1=np.array(data['u'].T)
y1=np.array(data['y'])
t1=np.array(data['t'].T)
para=(data['para'])



#order of polynomial for Hammerstein
p=3
na=4
nb=3
Ts=np.max(t1)/(len(t1)-1)
#---------------------------------------------------------------------------------------------------------------------------


#initial Values for the estimation
Pini=np.array(10**6*np.eye(na+p*(nb+1)))
pini=np.zeros(na+p*(nb+1))

sky=np.zeros(na)
sku=np.zeros(nb+1)
skT =np.zeros(na+p*(nb+1))
Sn=skT
#-----------------------------------------------------------------------------------------------------------------------------
#recursive algorithm

for k in range(0,len(t1)):
    pini,Pini,sku,sky,skT = Hammerstein(pini,Pini,sku,sky,y1,u1,skT,na,nb,k,p)


#---------------------------------------------------------------------------------------------------------------------------


#now its a little harder to calculate the system response since we cannot set up the transfer function as easily. 
#however we do have the parameter vector pini that we calculated through our recursion. Assume now that the input is known, but we 
#do not have knowledge of the ouput. So using yest=Sn*pini would not yield a satisfying solution for the ouput error since we assume we do not know the previous output.
#Assuming we have the input

sy=np.zeros(na)
su=np.zeros(nb+1)
yest=np.zeros(len(t1))

for k in range(0,len(t1)):
    #lets start at time step 0: we only have the input u in our skT vector
    su,sy,skT,yest=solveHammerstein(pini,y1,u1,sy,su,skT,k,yest,na,nb,p)


#---------------------------------------------------------------------------------------------------------------------------
#computing the error as follows: 

e=np.zeros(len(t1))
for i in range(0,len(t1)):
    e[i]=y1[i,0]-yest[i]



#---------------------------------------------------------------------------------------------------------------------------



plt.figure(figsize=(10,8))
# Plot the estimated output response
plt.subplot(3, 1, 1)  # 2 rows, 1 column, subplot 1
plt.plot(t1, yest, label='Estimated Output ')
plt.title('Estimated vs. Real Output training data')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid()

# Plot the real output
plt.subplot(3, 1, 2)  # 2 rows, 1 column, subplot 2
plt.plot(t1, y1, label='Real Output')
plt.xlabel('Time')
plt.ylabel('Output')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)  # 2 rows, 1 column, subplot 3
plt.plot(t1, e, label='estimation error')
plt.xlabel('Time')
plt.ylabel('Output error')
plt.legend()
plt.grid()
plt.tight_layout() 
plt.ylim(-0.1,0.1) # Adjust layout to prevent overlapping
plt.show()



