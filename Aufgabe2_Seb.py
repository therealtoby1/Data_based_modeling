from scipy.io import loadmat
from scipy.signal import freqz, dlti, dlsim
import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#---------------------------------------------------------------------------------------------------------------------------

#extracting the Data from the matlab file

data=loadmat(r'C:\KIT\Master_KIT\DMC\Seb_2\ex2_data.mat')
#print(data.keys())

x=data['X']
t=data['t']
u=data['u']




#we now create the snapshot matrices of the output y
X = x[:,:-1]
Xp = x[:,1:]
Uin = u[:,0:-1]
#---------------------------------------------------------------------------------------------------------------------------

Omega = np.vstack((X,Uin))

#single value decomposition of X
U,S,Vt = np.linalg.svd(Omega)
#order of approximation
p = np.linalg.matrix_rank(Omega)
r = 5

if r > p:
    r = p


U = U[:,0:r]
#Abspeichern der singular values für part 2
sings = S
S = np.diag(S)[0:r,0:r]
Vt = Vt[0:r , :]
#---------------------------------------------------------------------------------------------------------------------------
UT=U.T
V=Vt.T
Sinv=np.linalg.inv(S)

#number of inputs 
nb=2
#number of states (or in this case outputs)
nx=202
#after the SVD we compute the solution to our system by:
G=Xp@V@Sinv@UT
# splitting our system matrix G into [A,B] 
A = G[:,:nx]
B = G[:, nx:]
#---------------------------------------------------------------------------------------------------------------------------
#dimension reduction now with At=Uh* Xp V S^-1 U1* Uh =Uh* A U ,
#with Uh from the single value decomposition of the output space

Uh,Sh,Vth = np.linalg.svd(Xp)

Uh = Uh[:,0:r]
UhT = Uh.T

At = UhT@A@Uh
Bt = UhT@B
#---------------------------------------------------------------------------------------------------------------------------
#calculating the ouput in the reduced order system with the input u
x0 = X[:,0]
z = np.zeros((r,len(t.T)))
z[:,0] = UhT@x0

for i in range(1,len(t.T)):
    z[:,i] = At@z[:,i-1] + Bt@u[:,i]

#backtransformation to the original system order
xt=Uh@z






#--------------------------------------------------------------------------------------------------------------------------- 
#Generating the mesh


Zaxis=np.linspace(0, 1, len(x))
Z, T = np.meshgrid(Zaxis, t, indexing='ij')









#---------------------------------------------------------------------------------------------------------------------------



# Plotting
fig = plt.figure(figsize=(15, 5))

# Plot the real values
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(Z, T, x, cmap='viridis')
ax1.set_xlabel('Spatial Points')
ax1.set_ylabel('Time Points')
ax1.set_zlabel('Data Values')
ax1.set_title('Real Values')

# Plot the DMDc values
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(Z, T, xt, cmap='viridis')
ax2.set_xlabel('Spatial Points')
ax2.set_ylabel('Time Points')
ax2.set_zlabel('Data Values')
ax2.set_title('DMDc')

# Plot the error
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(Z, T, x-xt, cmap='viridis')
ax3.set_xlabel('Spatial Points')
ax3.set_ylabel('Time Points')
ax3.set_zlabel('Data Values')
ax3.set_title('Error')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()



#---------------------------------------------------------------------------------------------------------------------------

#part b) we have 204 eigenvalues
ratio=np.zeros(len(sings))
sumsings=np.sum(sings)
for i in range(0,len(sings)):
    ratio[i]=np.sum(sings[:i])/sumsings

Number_of_sings = np.arange(0, 204)
#---------------------------------------------------------------------------------------------------------------------------
plt.scatter(Number_of_sings[:10], ratio[:10])
plt.xlabel('#Singular values')
plt.ylabel('Ratio')
plt.title('Ratio Over Number of Singular values')
plt.grid(True)
plt.show()
