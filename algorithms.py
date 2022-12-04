import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def lower_bound_milos(D,r):
  """
  D - disimilarity matrix
  r - embedding dimension
  """
  n = D.shape[0]
  v = np.ones(n)
  v[n-1] = 1+np.sqrt(n)
  Q = np.eye(n) - (2/ (np.transpose(v) @ v)) * np.outer(v,v)
  tempD = (Q@D@Q)
  D_hat = tempD[0:n-1, 0:n-1]
  f = tempD[0:n-1, n-1]
  e = tempD[n-1, n-1]
  # possible error with the shape of U
  Lambda, U = np.linalg.eig(D_hat)
  # want the eigenvalues to be in ascending order
  idx = Lambda.argsort()  
  Lambda = Lambda[idx]
  U = U[:,idx]
  my_l = Lambda

  c = np.append(Lambda, e)
  negative_C2 = 0
  for i in range(0,n-1):
    if Lambda[i] > 0 or (i+1) > r:
      negative_C2 += Lambda[i]
      c[i]=0
  # number of cs not equal to 0
  E = np.sum(c!=0)
  sub = negative_C2/E

  for i in range(0,n-1):
    if c[n-i-2] != 0:
      if c[n-i-2] + sub <=0:
        c[n-i-2]+=sub
        E-=1
        negative_C2-=sub
      else:
        E-=1
        negative_C2 += c[n-2]
        sub = negative_C2/E
        c[n-2] = 0

  c[n-1] += sub
  negative_C2 -=sub
  D_hat = U @ np.diag(c[0:n-1]) @ np.transpose(U)
  block_ma = np.block([[D_hat, f.reshape((-1,1))], [f.reshape((1,-1)), c[n-1]]])
  output = Q @ block_ma @ Q
  output = 0.5 * (output + output.T)
  return output

def lower_bound(Dp,r):
    """
    Dp similarity matrix numpy array
    r int embedding dimension
    """ 
    Dp = torch.tensor(Dp, dtype=torch.float)
    n,n = Dp.shape
    v = torch.ones(n,1)
    v[n-1] += np.sqrt(n)
    Q = torch.eye(n) - 2*(v.mm(v.t()))/(v.square().sum())

    Dphat = Q.mm(Dp.mm(Q))

    l,v  = torch.symeig(Dphat[0:n-1,0:n-1], eigenvectors=True)
    Lambda = torch.clone(l) #want the eigenvalues to be in accending order. 

    s = 0
    E = n
    for i in range(n-1):
        if Lambda[i] > 0 or i > r-1:
            s += Lambda[i] 
            Lambda[i] = 0
            E = E-1

    sub = s/E

    for i in range(n-1):
        if Lambda[n-2-i] != 0:
            if Lambda[n-2-i] + sub <= 0 or i == 0:
                Lambda[n-2-i] += sub
                E -= 1
                s -= sub
            else:
                E -= 1
                s += Lambda[n-2-i]
                Lambda[n-2-i] = 0
                sub = s/E
                
    Dphat[n-1,n-1] += sub
    E -= 1
    s -= sub

    L = torch.diag(Lambda)

    Dphat[0:n-1,0:n-1] = v.mm(torch.diag(Lambda).mm(v.t()))
    output = Q.mm(Dphat.mm(Q))
    # make sure that the output is symmetric
    # floating point error can mess up the symmetry
    output = 0.5 * (output + output.T)
    return np.array(output)
  
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