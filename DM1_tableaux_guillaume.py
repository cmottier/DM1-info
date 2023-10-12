#%%
import numpy as np
from math import log

#%%
A=np.array([[0.28,0.43,0.21,0.07,0.01],
[0.4096,0.4096,0.1536,0.0256,0.0016],
[0.3164063,0.421875,0.2109375,0.046875,0.00390625],
[0.2401,0.4116,0.2646,0.0756,0.0081],
[0.1785063,0.384475,0.3105375,0.111475,0.01500625]])

#Question 1
def kull(p):
    if p==0.2:
        yp1= -A[0,0]*log(A[1,0]/A[0,0])-A[0,1]*log(A[1,1]/A[0,1])-A[0,2]*log(A[1,2]/A[0,2])-A[0,3]*log(A[1,3]/A[0,3])-A[0,4]*log(A[1,4]/A[0,4])
        return yp1
    elif p==0.25:
        yp2= -A[0,0]*log(A[2,0]/A[0,0])-A[0,1]*log(A[2,1]/A[0,1])-A[0,2]*log(A[2,2]/A[0,2])-A[0,3]*log(A[2,3]/A[0,3])-A[0,4]*log(A[2,4]/A[0,4])
        return yp2
    elif p==0.3:
        yp3= -A[0,0]*log(A[3,0]/A[0,0])-A[0,1]*log(A[3,1]/A[0,1])-A[0,2]*log(A[3,2]/A[0,2])-A[0,3]*log(A[3,3]/A[0,3])-A[0,4]*log(A[3,4]/A[0,4])
        return yp3
    elif p==0.35:
        yp4= -A[0,0]*log(A[4,0]/A[0,0])-A[0,1]*log(A[4,1]/A[0,1])-A[0,2]*log(A[4,2]/A[0,2])-A[0,3]*log(A[4,3]/A[0,3])-A[0,4]*log(A[4,4]/A[0,4])
        return yp4

print("Calcul des contrastes de Kullback-Leibler :")
print("pour p=0.2 : ", kull(0.2))
print("pour p=0.25 : ", kull(0.25))
print("pour p=0.3 : ", kull(0.3))
print("pour p=0.35 : ", kull(0.35))

print("Deuxième version :")
for i in range(1,5):
    y1=0
    for j in range(5):
        y1 -= A[0,j]*log(A[i,j]/A[0,j])
    print(y1)

'''
On a pour p=0.3, la valeur la plus petite des 4, et donc la loi binomiale B(4, p=0.3)
est celle qui approche le mieux Q au sens de Kullback-Leibler
'''

#Question 2
print("Calcul des distances du chi2 :")
for i in range(4):
    dchi2=0
    for j in range(5):
        dchi2 += (A[0,j]-A[i+1,j])**2/A[0,j]
    print(dchi2)
'''
Pour la distance du chi2, la valeur minimale est atteinte pour p=0.25,
donc la loi binomiale B(4,p=0.25) est celle qui approche le mieux Q au sens du chi2
'''


# %%
B=np.array([
[27.8,189.7,70.0,119.6,187.1,76.9],
[117.4,914.0,357.9,556.1,525.8,184.8],
[564.9,2638.5,1209.0,1429.5,1161.6,360.0],
[1353.7,3735.7,1840.7,1895.0,1507.8,256.2],
[1570.9,3486.4,1605.6,1880.9,1819.6,397.0],
[1271.6,2648.6,1285.9,1362.7,1300.4,180.5],
[24.2,146.0,56.2,89.8,138.0,43.4],
[79.2,645.6,258.4,387.2,378.7,128.7],
[315.7,1538.5,685.3,853.2,719.0,240.1],
[613.3,1750.4,834.7,915.7,755.9,123.7],
[476.0,865.5,449.5,416.0,329.8,58.3],
[1085.8,2133.9,1068.4,1065.5,987.9,130.1]])


#On enlève la colonne en double (merci Camille je l'avais pas vue)
B_extrait=np.concatenate((B[:,0].reshape(12,1),B[:,2:6]),axis=1)

#Effectif total
total = np.sum(B_extrait)
print(total)
#Calcul avec la probabilité uniforme
prob=B_extrait*1/total
print(prob)

#Somme de tous les jobs
for i in range(6):
    print("Jobs : ", np.sum(B_extrait[i,:])+np.sum(B_extrait[i+6,:]))

#Somme par tranche d'age
for i in range(5):
    print("Age : ", np.sum(B_extrait[:,i]))

#Somme par sexe
a=np.sum(B_extrait[0:6,0:5])
print("Femmes : ", a)
b=np.sum(B_extrait[6:12,0:5])
print("Hommes : ", b)
print(a+b, total)

#Question 1
'''
Le nombre de variables concernant la catégorie professionnelle est le plus élevé des trois
Alors on pourrait la conditionner par le sexe et la tranche d'âge
'''

#I(C,(A x S))
icas=0
for j in range(5):
    for i in range(6):
        icas += prob[i,j]*log(prob[i,j]/(np.sum(prob[0:6,j])*(np.sum(prob[i,:])+np.sum(prob[i+6,:]))))
    for i in range(6,12):
        icas += prob[i,j]*log(prob[i,j]/(np.sum(prob[6:12,j])*(np.sum(prob[i,:])+np.sum(prob[i-6,:]))))
print(icas)

#I(A,(C x S))
iacs=0

# %%
C=np.array([[2,6,8,5,1],[27,10,8,5,0]])
# %%
