import numpy as np

print_statistics(images, t_images, labels, t_labels):
    #number of training dataset
    len(labels)
    #number of test dataset
    len(t_labels)
    #number of class
    len(np.unique(labels))
    #number of instances per class on training
    for i in np.unique(labels):
        np.sum(labels==i)
    #number of instances per class on test dataset
    for i in np.unique(t_labels):
        np.sum(t_labels==i)
        
 def eksponensia(X, theta_k):
    # theta_k adalah theta di kelas k
    sumo = theta_k[0]
    for i in range(1,len(X.iloc[0])):
        sumo += np.sum(theta_k[i+1]*X.iloc[:,i])
    return np.exp(sumo)

def hypothesis(X, theta, K):
    ekspo = []
    for i in range(K):
        ekspo.append(eksponensia(X, theta[:,i]))
        sumo = np.sum(ekspo)
    hipo = ekspo/sumo
    return hipo

def SoftMaxMod(X, theta, theta_k, K):
    expot =eksponensia(X, theta_k)
    sumeks = 0
    for i in range(K):
        sumeks += eksponensia(X, theta[:,i])
    return expot/sumeks

def CostFunct(X, y, theta, K):
    m =  len(y)
    sumo = 0
    for i in range(m):
        sumK = 0
        for j in range(K):
            if y.iloc[i,0] == j:
                sumK += np.log(SoftMaxMod(X, theta, theta[:,j], K))
        sumo += sumK
    return -sumo

def GradDesc(X, y, theta, knya, K):
    n = len(y)
    sumo = []
    for i in range(n):
        if y.iloc[i,0] == knya:
            sumo.append(1 - SoftMaxMod(X, theta, theta[:,knya], K))
        else:
            sumo.append(0 - SoftMaxMod(X, theta, theta[:,knya], K))
    return -np.sum(sumo)

def Thetanos(X, y, theta, alpha, K):
    f = len(X.iloc[0])
    for i in range(K):
        for j in range(f):
            theta[j+1][i] = theta[j+1][i] - (alpha*GradDesc(X, y, theta, i, K))
            print("theta[{}][{}] = {}".format(j+1,i,theta[j+1][i]))
    return theta

def SoftLearn(X, y, theta, alpha, itr):
    cost = []
    for i in range(itr):
        theta = Thetanos(X, y, theta, alpha, np.unique(y))
        cost.append(CostFunct(X, y, theta, np.unique(y)))
    return cost, theta