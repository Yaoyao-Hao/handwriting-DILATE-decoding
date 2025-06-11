import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import math
from sklearn import metrics
from sklearn.decomposition import *
from sklearn.cluster import *
from sklearn.metrics import rand_score,fowlkes_mallows_score,adjusted_rand_score,adjusted_mutual_info_score
from sklearn.manifold import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def PCA_feature(sequence,n):
    pca = PCA(n_components=n)
    H = pca.fit_transform(sequence.T)
    return H.T

q=mpl.cm.get_cmap("gist_ncar", 27)
c_list=q(range(26))
if __name__=="__main__":
    
    show=True
    for loss in ["MSE","DILATE"]:#
        FMI=[]

        ARI=[]

        AMI=[]
        
        ACC=[]
        for _ in range(1):
            st=0
            en=150
            data=np.load('./result/'+loss+'/{0}_{1}/prediction.npy'.format(st,en),allow_pickle=True)
            data=data.T.reshape(-1,en-st,2).mean(2).T
            unsorting_spike=data
            label=np.concatenate([[i for _ in range(27)] for i in range(26)],dtype=np.int32)
            
            shuf=np.random.permutation(label.shape[0])
            unsorting_spike=unsorting_spike[:,shuf]
            label=label[shuf]
            
            tsne=TSNE(n_components=2, init='pca')
            spike_feature=tsne.fit_transform(unsorting_spike.T).T
            
            if show:
                color=['#'+('%06x'%((i+1)*599999))[2:] for i in range(26)]
                fig, axi1=plt.subplots(1,dpi=300)
                for i in range(26):
                    axi1.scatter(spike_feature.T[label==i, 0], spike_feature.T[label==i, 1],
                                marker='o',
                                s=8,
                                c=c_list[i])
                plt.savefig('C:/Users/24233/Desktop/result/3p5-1.pdf', dpi = 300)
                plt.show()
                plt.close()
            
            n_clusters=26
            cluster = HDBSCAN().fit(spike_feature.T)
            
            y_pred = cluster.labels_
            
            FMI_score = fowlkes_mallows_score(label,y_pred)
            
            ari=adjusted_rand_score(label, y_pred)  
            
            ami=adjusted_mutual_info_score(label, y_pred) 
            
            FMI.append(FMI_score)
            ARI.append(ari)
            AMI.append(ami)
            ACC.append((np.array(label)==np.array(y_pred)).sum()/len(label))
            
        FMI=np.array(FMI)
        ARI=np.array(ARI)
        AMI=np.array(AMI)
        ACC=np.array(ACC)
        # print(FMI.mean(),ARI.mean(),AMI.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(unsorting_spike.T, label, test_size=0.1, random_state=42)
    
        # Create and train a KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
    
        # Predictive Test Set
        y_pred = knn.predict(X_test)
    
        # Evaluating model performance
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")