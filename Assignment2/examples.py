import numpy as np

def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    kNum = 0

    # TODO: compute the values of prior for each class!
    # ==========================
    for idx,className in enumerate(classes):
        idx = np.where(labels == className)[0]
        #print(className)
        #print(W[idx])
        prior[kNum] = np.sum(W[idx])
        #prior[kNum] = idx.shape[0] / Npts
        kNum += 1
    # ==========================

    #print(prior)
    return prior

labels = np.array([1,0,2,0,1,0])
votes = np.array([1,1,2,0,1,2])
k = np.array([1,1,1,1,1,1])
print((labels==votes)*k)
W = np.array([0.1,0.1,0.2,0.3,0.1,0.2])
#computePrior(labels, W)
#labels2 = np.array([1, 2, 3])
#print((labels2).reshape(1,-1).shape)
