import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

data = pd.read_csv('./CC_GENERAL.csv')

if 'CUST_ID' in data.columns:
    X = data.drop('CUST_ID', axis=1)
else:
    X = data.copy()
    
# Handle missing values (using updated pandas ffill syntax)
X = X.ffill()

# 2. Preprocessing
# Scale features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalize data to transform data points into unit L2 norm
X_normalized = normalize(X_scaled)

# Convert the numpy array back to a pandas DataFrame for convenience
X_normalized = pd.DataFrame(X_normalized)


pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
# Generate linkage using Ward's minimum variance method
dend = shc.dendrogram(shc.linkage(X_principal, method='ward'))
plt.xlabel('Sample Index')
plt.ylabel('Euclidean Distances')
plt.show()


cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
cluster.fit(X_principal)

ac2 = AgglomerativeClustering(n_clusters = 2)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'], X_principal['P2'],
c = ac2.fit_predict(X_principal), cmap ='rainbow')
plt.show()

# k = 3
ac3 = AgglomerativeClustering(n_clusters = 3)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'],
X_principal['P2'],
c = ac3.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# k = 4
ac4 = AgglomerativeClustering(n_clusters = 4)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'],
X_principal['P2'],
c = ac4.fit_predict(X_principal), cmap ='rainbow')
plt.show()


# k = 5
ac5 = AgglomerativeClustering(n_clusters = 5)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'],
X_principal['P2'],
c = ac5.fit_predict(X_principal), cmap ='rainbow')
plt.show()

# k = 6
ac6 = AgglomerativeClustering(n_clusters = 6)
plt.figure(figsize =(6, 6))
plt.scatter(X_principal['P1'],
X_principal['P2'],
c = ac6.fit_predict(X_principal), cmap ='rainbow')
plt.show()

k = [2, 3, 4, 5, 6]
models = [ac2, ac3, ac4, ac5, ac6]

silhouette_scores = []

for model in models:
    labels = model.fit_predict(X_principal)
    
    # Silhouette score requires between 2 and n_samples - 1 unique labels
    if len(set(labels)) > 1:
        score = silhouette_score(X_principal, labels)
    else:
        score = 0.0  # Fallback if the model collapses into a single cluster
        
    silhouette_scores.append(score)
    
plt.figure(figsize=(8, 5))
plt.bar(k, silhouette_scores)
plt.xlabel('Number of Clusters', fontsize=16)
plt.ylabel('Silhouette Score S(i)', fontsize=16)
plt.title('Silhouette Scores for Agglomerative Clustering', fontsize=16)
plt.show()       
