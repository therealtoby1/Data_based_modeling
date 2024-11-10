import numpy as np

def ARX (pini,Pini,sku,sky,y1,u1,skT,na,nb,k):
    yk = np.array(y1[k])
    uk = u1[k]
    
    if k==0:
        sky[1:na]=sky[0:na-1]
        sky[0]=0     
    else:  
        sky[1:na]=sky[0:na-1]
        sky[0]=y1[k-1,0]                     #teil für y
                                 
                  
   
                                        #Teil für u
    sku[1:nb+1]=sku[0:nb]
    sku[0]=u1[k,0] 

    skT=np.hstack((sky.T,sku.T)).reshape(1,na+nb+1)
    sk=np.array(skT.T).reshape(na+nb+1,1)
    
    #estimate of parameter vector p:
    kk=(Pini@sk/(1+skT@Pini@sk)).reshape(na+nb+1,1)
    Pini=Pini-kk@skT@Pini
    pini=pini+kk@(yk-skT@pini)
    return pini,Pini,sku,sky,skT

def Hammerstein (pini,Pini,sku,sky,y1,u1,skT,na,nb,k,p):
    yk=y1[k,0] 
    uk=u1[k]
    
    if k==0:
        sky[1:na]=sky[0:na-1]
        sky[0]=0     
    else:  
        sky[1:na]=sky[0:na-1]
        sky[0]=y1[k-1,0]                     #teil für y
                                 
                  
   
                                        #Teil für u
    sku[1:nb+1]=sku[0:nb]
    sku[0]=u1[k,0] 
    sku_=sku
    for i in range(2,p+1):
        sku_=np.concatenate((sku_, sku**i))
    
    


    skT=np.hstack((sky.T,sku_.T)).reshape(1,na+p*(nb+1))
    sk=np.array(skT.T).reshape(na+p*(nb+1),1)
    
    #estimate of parameter vector p:
    kk=(Pini@sk/(1+skT@Pini@sk)).reshape(na+p*(nb+1),1)
    Pini=Pini-kk@skT@Pini
    pini=pini+kk@(yk-skT@pini)
    #if k>0: 
    #    Sn=np.vstack((Sn,skT))
    return pini,Pini,sku,sky,skT
def solveHammerstein (pini,y1,u1,sy,su,skT,k,yest,na,nb,p):
    #lets start at time step 0: we only have the input u in our skT vector
   
    su[1:nb+1] = su[0:nb]
    su[0]      = u1[k,0]
    su_ = su

    for i in range(2,p+1):
        su_ = np.concatenate((su_,su**i))
    

    skT = np.hstack((sy.T,su_.T))

    #key step to get the estimated y vector
    yest[k] = skT@pini
    
    #updating our sy vector
    sy[1:na] = sy[0:na-1]
    sy[0] = yest[k]
    return su,sy,skT,yest
