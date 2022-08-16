import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import silhouette_score,silhouette_samples
from matplotlib import cm

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
mayo_inst=data[data.Mes=='Mayo']
julio_inst=data[data.Mes=='Julio']
atributos=data.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL'])
#Graficar grupos 
pca = PCA(2)
pca.fit(mayo_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
data_mayo = pca.transform(mayo_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
data_julio = pca.transform(julio_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
# plt.scatter(data_mayo[:, 0], data_mayo[:, 1], c="blue")
# plt.scatter(data_julio[:, 0], data_julio[:, 1], c="green")
# plt.show()
#Prediccion KMeans   
kmeans=KMeans(n_clusters=2,max_iter=10)
kmeans.fit(atributos)
pca = PCA(2)
pca.fit(atributos)
mes_data = pca.transform(atributos)
mes_data = np.c_[mes_data, data["Mes"]]
centers = pca.transform(kmeans.cluster_centers_)
primer_mes = mes_data[kmeans.labels_ == 0]
segundo_mes = mes_data[kmeans.labels_ == 1]
clase=[]
for i in data["Mes"]:
    if i== "Mayo":
        clase.append(1)
    else:
        clase.append(0)
print("Confusion Matrix:\n",confusion_matrix(clase,kmeans.labels_))
print(classification_report(clase,kmeans.labels_))
plt.scatter(primer_mes[:, 0], primer_mes[:, 1], c="yellow")
plt.scatter(segundo_mes[:, 0], segundo_mes[:, 1], c="orange")
plt.scatter(centers[:, 0], centers[:, 1], c="black")
plt.show()

#Encontrar la mejor k con elbow method
elbow = []
num_clusters = range(1,4)
for i in num_clusters:
    clustering = KMeans(n_clusters=i, init='k-means++', random_state=42)
    clustering.fit(atributos)
    elbow.append(clustering.inertia_)
    
plt.plot(num_clusters, elbow)
plt.title('Elbow curve')
plt.show()
print("ConfusionMatrix",confusion_matrix(clase,kmeans.labels_))
print(classification_report(clase,kmeans.labels_))

#Silhouete analysis
y_km=kmeans.fit_predict(atributos)
cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
si_score=silhouette_score(atributos,kmeans.labels_)
# print("Silhouette Score: %.3f" % si_score)
silhouette_vals=silhouette_samples(atributos,y_km,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.0)
    y_ax_lower+=len(c_silhouette_vals)
silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.yticks(yticks,cluster_labels+1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette Coefficients")
plt.show()

#Graficar grupos
pca.fit(es_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
plain_es = pca.transform(es_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
plain_ep = pca.transform(ep_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
plain_el = pca.transform(el_inst.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL']))
plt.scatter(plain_es[:, 0], plain_es[:, 1], c="Pink")
plt.scatter(plain_ep[:, 0], plain_ep[:, 1], c="peru")
plt.scatter(plain_el[:, 0], plain_el[:, 1], c="Green")
plt.show()
#Prediccion KMeans   

atributos=data.get(['MS', 'MP', 'ML', 'EsS', 'EsP', 'EsL'])
kmeans = KMeans(n_clusters=3)
kmeans.fit(atributos)
pca = PCA(2)
pca.fit(atributos)
cluster_data = pca.transform(atributos)
cluster_data = np.c_[cluster_data, data["Enfo"]]
centers = pca.transform(kmeans.cluster_centers_)
primer_cluster = cluster_data[kmeans.labels_ == 0]
segundo_cluster = cluster_data[kmeans.labels_ == 1]
tercer_cluster = cluster_data[kmeans.labels_ == 2]
clase=[]
for i in data["Enfo"]:
    if i== "ES":
        clase.append(0)
    elif i == "EL":
        clase.append(1)
    else:
        clase.append(2)
print("Confusion Matrix:\n",confusion_matrix(clase,kmeans.labels_))
print(classification_report(clase,kmeans.labels_))
plt.scatter(primer_cluster[:, 0], primer_cluster[:, 1], c="Red")
plt.scatter(segundo_cluster[:, 0], segundo_cluster[:, 1], c="Blue")
plt.scatter(tercer_cluster[:, 0], tercer_cluster[:, 1], c="Brown")
plt.scatter(centers[:, 0],centers[:, 1], c="Green")
plt.show()

#Prediccion Hierarchy Meses 
hierch = AgglomerativeClustering(n_clusters=2)
pca = PCA(2)
pca.fit(atributos)
mes_data = pca.transform(atributos)
model = hierch.fit_predict(mes_data)
mes_data = np.c_[mes_data, data["Mes"]]
cluster0 = mes_data[hierch.labels_ == 0]
cluster1 = mes_data[hierch.labels_ == 1]
# plt.scatter(cluster0[:, 0], cluster0[:, 1], c="Yellow")
# plt.scatter(cluster1[:, 0], cluster1[:, 1], c="Green")
clase=[]
for i in data["Mes"]:
    if i== "Mayo":
        clase.append(1)
    else:
        clase.append(0)
print("Confusion Matrix:\n",confusion_matrix(clase,kmeans.labels_))
print(classification_report(clase,hierch.labels_))
plt.show()
#Silhouete analysis
y_km=hierch.fit_predict(atributos)
cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
si_score=silhouette_score(atributos,hierch.labels_)
print("Silhouette Score: %.3f" % si_score)
silhouette_vals=silhouette_samples(atributos,y_km,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.0)
    y_ax_lower+=len(c_silhouette_vals)
silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.yticks(yticks,cluster_labels+1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette Coefficients")
plt.show()


#Prediccion Hierarchy Clases   


#Silhouete analysis
y_km=hierch.fit_predict(atributos)
cluster_labels=np.unique(y_km)
n_clusters=cluster_labels.shape[0]
si_score=silhouette_score(atributos,hierch.labels_)
#print("Silhouette Score: %.3f" % si_score)
silhouette_vals=silhouette_samples(atributos,y_km,metric='euclidean')
y_ax_lower,y_ax_upper=0,0
yticks=[]
for i,c in enumerate(cluster_labels):
    c_silhouette_vals=silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper+=len(c_silhouette_vals)
    color=cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower,y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.0)
    y_ax_lower+=len(c_silhouette_vals)
silhouette_avg=np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color='red',
            linestyle='--')
plt.yticks(yticks,cluster_labels+1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette Coefficients")
plt.show()