import numpy as np

def etiquetar(I, conectividad=4):
    imgExit = np.zeros(shape=I.shape)
    
    equivalencia =[]
    nLabel = int(1)
    for i in range(1,I.shape[0]):
        for j in range(1,I.shape[1]):
            
            A = imgExit[i-1,j-1]
            B = imgExit[i-1,j]
            C = imgExit[i,j-1]
            M = I[i,j]
            
            if M==0:
                if A>0:
                    imgExit[i,j]= A
                elif (B>0 and C==0):
                    imgExit[i,j]= B
                elif (B==0 and C>0):
                    imgExit[i,j]= C
                elif(B>0 and C>0):
                    imgExit[i,j]=C
                    imgExit[imgExit==C]=B
                else:
#                    
                    imgExit[i,j] = nLabel
                    nLabel = int(nLabel+1)
                    
            else:
                if(B>0 and C>0 and B!=C):
                    imgExit[imgExit==C]=B
#                     equivalencia.append((C,B))
            
    
    return imgExit