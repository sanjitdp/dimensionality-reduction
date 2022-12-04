import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.random import RandomState
import os
from PIL import Image
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
import sklearn
from mnist import MNIST
import matplotlib.pyplot as plt
import seaborn as sns



def get_dataset(name, sample_percent = 1.0, shuffle=False, normalize=False, dis_mat=False, perturbation="noise"):
  """
    Downloads the dataset, flattens it,
    and returns points and labels
  Args:
    name string 
      name of the dataset, choose from the listed
    sample_percent float in range [0.0, 1.0]
      if you are testing a slow algrithm with sample percent you can take
      a part of the dataset
    shuffle bool
      if True will shuffle the dataset
    normalize bool
      if True will normalize using scaler
    dis_mat
      if true returns a dissimilarity matrix instead of points
    perturbation string
      only applied if dis_mat=True, the way to perturn disimilarity matrix
      can be one of three values "noise", "isomap", "missing" for 3 methods
  Return:
    if dis_mat=False
    input numpy.ndarray float64
      2D array of shape (n, m) where n is the number of points and m the number of features
    labels numpy.ndarray
      1D array of shape (n) where n is the number of points
    if dis_mat=True
    disimilarity matrix numpy.ndarray float64
      2D array of shape (n, n) where n is the number of points and each entry (i,j) represent the distance betwee x_i and x_j
    labels numpy.ndarray
      1D array of shape (n) where n is the number of points
  """
  input = None
  labels = None
  if name == "mnist":
    mndata = MNIST('datasets')
    train_X, train_y = mndata.load_training()
    test_X, test_y = mndata.load_testing()
    input = np.vstack((train_X,test_X))
    input = input.reshape((input.shape[0],-1))
    labels = np.append(train_y, test_y)
    print(input.shape)
    print(labels.shape)

  elif name == "s_curve":
    rng = RandomState(0)
    n_samples = 1500

    S_points, S_color = sklearn.datasets.make_s_curve(n_samples, random_state = rng)
    input = S_points
    labels = S_color
  elif name == "swiss_roll":
    rng = RandomState(0)
    n_samples = 1500

    swiss_roll_points, swiss_roll_color = sklearn.datasets.make_swiss_roll(n_samples, random_state=rng)
    input = swiss_roll_points
    labels = swiss_roll_color
  elif name == "coil20":
    folder_name = "datasets/coil-20"
    input = []
    labels = []
    for img_name in os.listdir(folder_name):  
      img_path = os.path.join(folder_name,img_name)
      input.append(np.array(Image.open(img_path)))
      temp = img_name.split("__")[0]
      obj_id = temp.split('j')[1]
      labels.append(int(obj_id))
    input = np.array(input)
    input = input.reshape((input.shape[0],-1))
    labels = np.array(labels)
  elif name == "coil100":
    folder_name = "datasets/coil-100"
    input = []
    labels = []
    for img_name in os.listdir(folder_name):  
      img_path = os.path.join(folder_name,img_name)
      input.append(np.array(Image.open(img_path)))
      temp = img_name.split("__")[0]
      obj_id = temp.split('j')[1]
      labels.append(int(obj_id))
    input = np.array(input)
    input = input.reshape((input.shape[0],-1))
    labels = np.array(labels)
  elif name=="google_news":
    input = np.load("word2vec-google-news-300.model.vectors.npy")
    labels= np.ones(input.shape[0])
  # if not nan
  ds_len = input.shape[0]
  if shuffle:
    idx = np.random.permutation(len(input))
    input = input[idx]
    labels=labels[idx]
  if normalize:
    scaler = StandardScaler()
    input = scaler.fit_transform(input)
  input = input[0:int(sample_percent*ds_len) ]
  labels = labels[0:int(sample_percent*ds_len) ]
  if dis_mat:
    if perturbation=="noise":
      D = pairwise_distances(input,input)
      n = D.shape[0]
      # mean 0, sd=1
      b = np.random.normal(0, 1, (n,n)) * 10 
      # make it symmetric
      b = 0.5 * (b + b.T)
      # fill diagonals with zero
      np.fill_diagonal(b, 0)
      D_pert = D + b
      return D_pert**2, labels
    elif perturbation=="missing":
      p=0.5
      n,m = input.shape
      Q = torch.rand(n,m) > p
      D = torch.zeros(n,n)
      for i in tqdm(range(n)):
          for j in range(i):
              D[i,j] = (Q[i,:]*Q[j,:]*(input[i,:]-input[j,:])).square().sum()
              D[j,i] = D[i,j]
      return np.array(D)**2, labels
    elif perturbation=="isomap":
      D = np.load("datasets/mnist-isomap.npy")
      return D, labels  
    else:
      print("RETURNING POINTS NOT SIM MATRIX")
      return input, labels
  else:
    return input,labels

def make_plot(embedding, labels, one_color=False, palette="tab10", marker_size=0.2, eigv1 = 0, eigv2 = 1):
  ax=None
  # if not nan
  if one_color:
    ax = sns.scatterplot(x=embedding[:,0],
                y=embedding[:,1],
                legend='full',
                s=marker_size)
  else:
    dif_col = len(np.unique(labels))
    ax = sns.scatterplot(x=embedding[:,eigv1],
                    y=embedding[:,eigv2],
                    hue=labels,
                    legend='full',
                    palette=sns.color_palette(palette, dif_col),
                    s=marker_size)
    sns.move_legend(ax, "upper right", markerscale=2.0)
  plt.gca().set_aspect('equal', 'datalim')
  plt.gcf().set_size_inches((10, 10))
  return ax
def procrustes(X, Y):
    """
        Ya = procrustes(X, Y)

    Returns Ya = alpha * (Y - muY) * Q + muX, where muX and muY are the m x n
    matrices whose rows contain copies of the centroids of X and Y, and alpha
    (scalar) and Q (m x m orthogonal matrix) are the solutions to the Procrustes
    + scaling problem

    Inputs: `X` and `Y` are m x n matrices

    Output: `Ya` is an m x n matrix containing the Procrustes-aligned version
    of Y aligned to X and Q the optimal orthogonal matrix

    min_{alpha, Q: Q^T Q = I} |(X - muX) - alpha * (Y - muY) Q|_F
    """
    muX = np.mean(X, axis=0)
    muY = np.mean(Y, axis=0)
    
    X0 = X - muX 
    Y0 = Y - muY 
    # Procrustes rotation
    U, _, V = np.linalg.svd(np.transpose(X0) @ Y0, full_matrices=False)
    V=np.transpose(V)
    Q = V @ np.transpose(U)
    # Optimal scaling
    alpha = np.trace(np.transpose(X0) @ Y0 @ Q) / np.trace(np.transpose(Y0) @ Y0)

    # Align data
    Ya = alpha * (Y0 @ Q) + muX

    return Ya
  
# Assume input in distance squared torch tensor.
def mds(D, d, center = True, align = True):
    n = D.shape[0]

    P = torch.eye(n) - torch.ones(n,n)/n
    K = 0.5*P.mm(D.mm(P))
    K = (K+K.t())/2

    e,V = torch.symeig(K, eigenvectors=True)
    e = -1*e
  
    if d > (e > 0).sum():
        print("Using negative eigenvalues")
  
    X = V[:,0:d].mm(torch.diag(e[0:d].sqrt()))

    return X
  
def make_NN(input_size):
    classifier = torch.nn.Sequential(torch.nn.Linear(input_size,100),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(100,100),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(100,10),
                                     torch.nn.LogSoftmax(dim = 1))
    return classifier

def train(model, X, y):
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    for i in range(500):
        y_pred = model(X)
        loss = torch.nn.functional.nll_loss(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    return model

def test_accuracy(model, X, y):
    y_pred = torch.argmax(model(X), dim = 1)
    return (y_pred == y).sum()/y.shape[0]