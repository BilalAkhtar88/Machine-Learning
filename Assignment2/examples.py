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
    # Iterate over both index and value
    for idx,className in enumerate(classes):
        idx = np.where(labels == className)[0]
        prior[kNum] = idx.shape[0] / Npts
        kNum += 1

    return prior

labels = np.array([1,0,2,0,1,2,0])
computePrior(labels)
labels2 = np.array([1, 2, 3])
print(np.shape(labels2))
