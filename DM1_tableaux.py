#%%
import numpy as np
from math import log

#%%

""" Exercice 1 """

A=np.array([[0.28,0.43,0.21,0.07,0.01],
[0.4096,0.4096,0.1536,0.0256,0.0016],
[0.3164063,0.421875,0.2109375,0.046875,0.00390625],
[0.2401,0.4116,0.2646,0.0756,0.0081],
[0.1785063,0.384475,0.3105375,0.111475,0.01500625]])

def K(i):
    S=0
    for j in range(5):
        S+=log(A[0,j]/A[i,j])*A[0,j]
    return S

print(K(1),K(2),K(3),K(4))

#On garde donc la troisième (contraste minimal)

def Khi2(i):
    S=0
    for j in range(5):
        S+=((A[0,j]-A[i,j])**2)/A[0,j]
    return S

print(Khi2(1),Khi2(2),Khi2(3),Khi2(4))

#On garde donc la deuxième (distance minimale)

# %%

"""Exercice 2"""

#Question : quelles colonnes garder ? 30-49 ou 30-39 et 40-49 ???

B=np.array([[27.8,189.7,70,119.6,187.1,76.9],
[117.4,914,357.9,556.1,525.8,184.8],
[564.9,2638.5,1209,1429.5,1161.6,360],
[1353.7,3735.7,1840.7,1895,1507.8,256.2],
[1570.9,3486.4,1605.6,1880.9,1819.6,397],
[1271.6,2648.6,1285.9,1362.7,1300.4,180.5],
[24.2,146,56.2,89.8,138,43.4],
[79.2,645.6,258.4,387.2,378.7,128.7],
[315.7,1538.5,685.3,853.2,719,240.1],
[613.3,1750.4,834.7,915.7,755.9,123.7],
[476,865.5,449.5,416,329.8,58.3],
[1085.8,2133.9,1068.4,1065.5,987.9,130.1]])

#Enlève la colonne redondante :
B_extrait=np.concatenate((B[:,0].reshape(12,1),B[:,2:6]),axis=1)

#Population totale et calcul des probabilités :
tot=np.sum(B_extrait)
Proba=B_extrait*1/tot

#I(C,(AxS))

ICAS=0
for j in range(5):
    for i in range(6):
        ICAS+=Proba[i,j]*log(Proba[i,j]/((np.sum(Proba[i,:])+np.sum(Proba[i+6,:]))*np.sum(Proba[0:6,j])))
    for i in range(6,12):
        ICAS+=Proba[i,j]*log(Proba[i,j]/((np.sum(Proba[i,:])+np.sum(Proba[i-6,:]))*np.sum(Proba[6:12,j])))

#I(S,(AxC)):

ISAC=0
for j in range(5):
    for i in range(6):
        ISAC+=Proba[i,j]*log(Proba[i,j]/(np.sum(Proba[0:6,:])*(Proba[i,j]+Proba[i+6,j])))
    for i in range(6,12):    
        ISAC+=Proba[i,j]*log(Proba[i,j]/(np.sum(Proba[6:12,:])*(Proba[i,j]+Proba[i-6,j])))

#I(A,(CxS)) :

IACS=0
for j in range(5):
    for i in range(12):
        IACS+=Proba[i,j]*log(Proba[i,j]/(np.sum(Proba[:,j])*np.sum(Proba[i,:])))

print("ICAS",ICAS,"ISAC",ISAC,"IACS",IACS)

#On préfère I(C,(AxS)) (information maximale)

#I(A,S)

IAS=0
for j in range(5):
    IAS+=np.sum(Proba[0:6,j])*log(np.sum(Proba[0:6,j])/(np.sum(Proba[:,j])*np.sum(Proba[0:6,:])))
    IAS+=np.sum(Proba[6:12,j])*log(np.sum(Proba[6:12,j])/(np.sum(Proba[:,j])*np.sum(Proba[6:12,:])))

#I(A,C)

IAC=0
for j in range(5):
    for i in range(6):
        IAC+=(Proba[i,j]+Proba[i+6,j])*log((Proba[i,j]+Proba[i+6,j])/(np.sum(Proba[:,j])*(np.sum(Proba[i,:])+np.sum(Proba[i+6,:]))))

#I(S,C)

ISC=0
for i in range(6):
    ISC+=np.sum(Proba[i,:])*log(np.sum(Proba[i,:])/(np.sum(Proba[0:6,:])*(np.sum(Proba[i,:])+np.sum(Proba[i+6,:]))))
for i in range(6,12):
    ISC+=np.sum(Proba[i,:])*log(np.sum(Proba[i,:])/(np.sum(Proba[6:12,:])*(np.sum(Proba[i,:])+np.sum(Proba[i-6,:]))))

print("IAS",IAS,"IAC",IAC,"ISC",ISC)

#On segmente dans un premier temps par la catégorie (IAC+ISC maximale)
#Deuxième variable sans importance car il ne reste plus que deux variables : I(A,S)=I(S,A)
#Lien avec la question précédente ??? 

#%%

"""Exercice 3"""

C=np.array([[2,6,8,5,1],[27,10,8,5,0]])
tot=np.sum(C)

#Regroupement par classes de X

X=1/tot*np.sum(C,axis=0)

"""Deux classes : """

def HX(j):
    return -(np.sum(X[0:j])*log(np.sum(X[0:j]))+np.sum(X[j:])*log(np.sum(X[j:])))

# for j in range(1,5):
#     print(HX(j))    

#On coupe en deux classes : =0 ou !=0

"""Trois classes : """

def HXbis(j,k):
    return -(np.sum(X[0:j])*log(np.sum(X[0:j]))
    +np.sum(X[j:k])*log(np.sum(X[j:k]))
    +np.sum(X[k:])*log(np.sum(X[k:]))
    )

# for j in range(1,4):
#     for k in range(j+1,5):
#         print(j,k,HXbis(j,k))

#On coupe en trois classes : =0, <0.5, >0.5

#Info mutuelle de Y avec un regroupement en deux classes :

XY=1/tot*C

def IXY(j):
    return (np.sum(XY[0,0:j])*log(np.sum(XY[0,0:j])/(np.sum(XY[0,:])*np.sum(XY[:,0:j])))
    +np.sum(XY[1,0:j])*log(np.sum(XY[1,0:j])/(np.sum(XY[1,:])*np.sum(XY[:,0:j])))
    +np.sum(XY[0,j:])*log(np.sum(XY[0,j:])/(np.sum(XY[0,:])*np.sum(XY[:,j:])))
    +np.sum(XY[1,j:])*log(np.sum(XY[1,j:])/(np.sum(XY[1,:])*np.sum(XY[:,j:]))))

for j in range(1,4):
    print(IXY(j)) 
print(np.sum(XY[0,0:j])*log(np.sum(XY[0,0:j])/(np.sum(XY[0,:])*np.sum(XY[:,0:j])))
    +np.sum(XY[1,0:j])*log(np.sum(XY[1,0:j])/(np.sum(XY[1,:])*np.sum(XY[:,0:j])))
    +np.sum(XY[0,j:])*log(np.sum(XY[0,j:])/(np.sum(XY[0,:])*np.sum(XY[:,j:]))))

#Pour la prédiction de Y, le meilleur regroupement est encore par la première classe.
# %%
