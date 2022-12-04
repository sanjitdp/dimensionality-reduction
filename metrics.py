from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
import numpy as np
from utils import procrustes

def knn_accuracy(embedding, labels, k=100, num_folds=10):
  """
  Computes k nearest neigbour classifier accuracy
  of the dimensionality reduction method
  Args
    embedding numpy.ndarray float64
      2D array of shape (n,2) where n is the number of embedded points.
      this array is a dimension reduced dataset 
      i.e. it is the output of dimensionality reduction algorithm
    labels numpy.ndarray int
      1D array oint shape=[n] where n is the number of input points 
    k int 
      number of neighbours to use for knn classifier
  """
  acc = []
  for i in range(num_folds):
    X_train, X_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.2)
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    predictions = neigh.predict(X_test)
    acc.append(accuracy_score(predictions, y_test))
  acc=np.array(acc)
  return np.mean(acc), np.std(acc)

def stability(embeddings1, embeddings2):
  """
  Stability is a metric of the algorithm on some dataset
  First we compute embeddings1 by subsampling 20% of the initial high dimensional dataset
  """
  # normalize the data
  embeddings1 = embeddings1/np.mean(np.linalg.norm(embeddings1,axis=1))
  embeddings2 = embeddings2/np.mean(np.linalg.norm(embeddings2,axis=1))
  embeddings2 = procrustes(embeddings1, embeddings2)
  return np.mean(np.sqrt(np.sum((embeddings1-embeddings2)**2, axis=1)))

""" EXAMPLE USAGE OF STABILITY 
# it should be computed 10 times and averaged over those 10 values

# SAMPLE 20% of the dataset and embed it
reducer = umap.UMAP(min_dist=0.05, n_neighbors=20)
input1, labels1 = get_dataset("coil20",sample_percent=0.2)
embedding1 = reducer.fit_transform(input1)

# embed the whole dataset and sample 20%
reducer = umap.UMAP(min_dist=0.05, n_neighbors=20)
input2, labels2 = get_dataset("coil20")
embedding2 = reducer.fit_transform(input2)
embedding2 = embedding2[0:int(0.2*len(embedding2)) ]
labels2 =  labels2[0:int(0.2*len(labels2)) ]
# compute the metric
stability(embedding1, embedding2)
"""
import numpy as np
def stress(X, Y):
  """
  X - input 
  Y - embeddings
  """
  D = pairwise_distances(X)
  D_hat = pairwise_distances(Y)
  return np.linalg.norm(D - D_hat) ** 2

def strain(X, Y):
  """
  X - input
  Y - embeddings
  """
  n = X.shape[0]
  D = pairwise_distances(X)
  I = np.eye(n)
  J = np.ones((n,n))
  V = I - 1/n * J
  return np.linalg.norm( (Y@Y.T) - (-V @ D @ V)/2 ) ** 2