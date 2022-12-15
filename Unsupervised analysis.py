import utils 
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans, SpectralClustering, AgglomerativeClustering, Birch
from tsne import tsne 

dataset =utils.load_data(path="./dataset")
data = dataset.reshape(179, 128*72*3)

# Metaparametre

perplexity = 21
n_clusters = 9
n_init = 100

# Coordonees des images par t-SNE

tsnecoord = tsne(data, perplexity=perplexity)

# Clustering sur le resultat

clusters = KMeans(n_clusters=n_clusters, n_init=n_init).fit_predict(data)
# clusters = SpectralClustering(n_clusters=n_clusters, n_init=n_init).fit_predict(data)
# clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(data)
# clusters = Birch(n_clusters=n_clusters).fit_predict(data)

# Sortie graphique
   
colors = utils.get_colors([i for i in range(n_clusters)])
for  i in range(n_clusters):
    plt.scatter(tsnecoord[clusters == i, 0], tsnecoord[clusters == i, 1], c = colors[i], marker='o')
plt.savefig('results/Perplexity_' + str(perplexity) + '_clusters_' + str(n_clusters) + '.png', dpi=300, bbox_inches='tight')
plt.show()
leafgraph = utils.imscatter(tsnecoord[:,0], tsnecoord[:,1], dataset)
plt.savefig('results/Perplexity_' + str(perplexity) + '_clusters_' + str(n_clusters) + '_leaves.png', dpi=300, bbox_inches='tight')