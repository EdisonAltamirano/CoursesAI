from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
may_csv="mayo1.csv"
july_csv="julio.csv"
may=pd.read_csv(may_csv)
#print(may)
july = pd.read_csv(july_csv)
# print(july.columns)
# print([str(i) for i in range(1, 43)])
#July
july.columns=['Marca temporal', 'Sexo', 'Carrera']+[str(i) for i in range(1, 43)]+['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL', 'ES', 'EP', 'EL', 'Enfo']
july.drop(["Marca temporal"]+[str(i) for i in range(1, 43)],inplace=True,axis=1)
july.insert(len(july.columns),"Mes","Julio")
#May
may.columns=['id', 'Sexo', 'Carrera']+[str(i) for i in range(1, 43)]+['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL', 'ES', 'EP', 'EL', 'Enfo']
may.drop(["id"]+[str(i) for i in range(1, 43)],inplace=True,axis=1)
may.insert(len(may.columns),"Mes","Mayo")
data = pd.concat([may,july]) 
es_inst=data[data.Enfo=='ES']
ep_inst=data[data.Enfo=='EP']
el_inst=data[data.Enfo=='EL']
data["Carrera"] = data["Carrera"].str.upper()
replace_dict={}
for index,j in enumerate(data):
    count =0
    for i in set(data.iloc[:, index]):
        if(type(i)==str):
            replace_dict[i]=count
            count+=1
sexo_class=data.groupby('Sexo').size().shape[0]
carrera_class=data.groupby('Carrera').size().shape[0]
numeric_df = data.iloc[:,:].replace(replace_dict)
classes=np.array(numeric_df.get(['Carrera']))
atributos=numeric_df.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']).to_numpy()
pca = PCA(2)
pca.fit(atributos)
pca_data = pca.transform(atributos)
DBSCAN_cluster = DBSCAN(eps=1, min_samples=2).fit(atributos) 
labels=DBSCAN_cluster.labels_
no_clusters = len(np.unique(labels) )
no_noise = np.sum(np.array(labels) == -1, axis=0)
asignar=[]
fig = plt.figure()
colores=['blue','red','green','blue','cyan','yellow','orange','black','pink','brown','purple','dimgray','gray','white', 'brown','chocolate','peru','linen','bisque','darkorange','burlywood','tan','orange','wheat','gold','olivedrab','yellowgreen','lawngreen','lime','turquoise','darkblue']
print("Sexo: "+str(sexo_class)+" clases")
print("Carrera: "+str(carrera_class)+" clases")

analyse_dict={}
for i in replace_dict:
    analyse_dict[i]=colores[replace_dict[i]]
print(analyse_dict)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)
for row in classes:
    asignar.append(colores[row[0]])
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=asignar)
plt.show()
clusters = []
print(np.unique(labels))
for cluster_id in labels:
    clusters.append(pca_data[DBSCAN_cluster.labels_ == cluster_id])
# for cluster in clusters:
#     plt.scatter(cluster[:, 0], cluster[:, 1], alpha=0.75)
# plt.show()

# print(atributos)
# atributos=atributos.to_numpy()
# # Generate scatter plot for training data
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(atributos[:,0], atributos[:,1], c=colors, marker="o", picker=True)
# plt.title('Two clusters with data')
# plt.xlabel('Axis X[0]')
# plt.ylabel('Axis X[1]')
# plt.show()
