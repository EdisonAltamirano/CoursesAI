from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

Archivo=open('Datos1.csv','r')
NomClase=["yes","no"]

#Archivo=open('Datos2.csv','r')
#NomClase=["Iris-virginica","Iris-versicolor","Iris-setosa"]
datos=[]
cont=1
Clases=[]

for x in Archivo.readlines():
    if len(x)>2:
        if cont==1:
            Atributos=x.replace('\n','').split(',')
            cont+=1
        else:
            atrib=x.replace('\n','').split(',')
            datos.append(list(map(float,atrib[:-1])))
#print(datos)
Archivo.close()

#print(datos)
kmeans = KMeans(n_clusters=2, max_iter=10).fit(datos)
print(kmeans.labels_)

print(kmeans.cluster_centers_)

#Ejemplo=kmeans.predict([[5, 2], [4, 0], [4, 1]])
Ejemplo=kmeans.predict([[64,89], [50,90]])
print(Ejemplo)
