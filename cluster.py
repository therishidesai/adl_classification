import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

class cluster():
    def __init__(self, sklearn_load_ds, targets, random):
        data = sklearn_load_ds
        X = pd.DataFrame(data)
        self.X_data = sklearn_load_ds
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, targets, test_size=0.3, random_state = random)
        self.random = random
        
    def classify(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy, confusion_matrix(self.y_test, y_pred)
        #print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))
        #print(confusion_matrix(self.y_test, y_pred))

    def Kmeans(self, output='add', tune=1, k=1):
        n_clusters = int(k*tune)
        clf = KMeans(n_clusters = n_clusters, random_state = self.random, n_jobs=2)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        if output == 'a':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
        elif output == 'r':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        return self

    #def find_k(self, max_k = 50):
    #    for i in range(max_k):
            
    def Kmeans_test(self, k = 1):
        n_clusters = k
        clf = KMeans(n_clusters = n_clusters, random_state = self.random, n_jobs = 2)
        clf.fit(self.X_train)
        return sum(np.min(cdist(self.X_train, clf.cluster_centers_, 'euclidean'), axis = 1))
        
#32x3 data setup
def parse_data():
    frames = []
    files = {"Brush_teeth":[], "Climb_stairs":[], "Comb_hair":[], "Descend_stairs":[], "Drink_glass":[], "Eat_meat":[], "Eat_soup":[], "Getup_bed":[], "Liedown_bed":[], "Pour_water":[], "Sitdown_chair":[], "Standup_chair":[], "Use_telephone":[], "Walk":[]}
    for k, v in files.items():
        dir_name = "HMP_Dataset/" + k
        files[k] = os.listdir(dir_name)

    dfs = {}
    sizes = []
    for k, v in files.items():
        dir_name = "HMP_Dataset/"+k
        frames = []
        for f in v:
            fname = dir_name + "/" + f
            vals = pd.read_table(fname, sep='\s+').values
            rows = vals.shape[0]
            chunks = int(rows/32)
            chunk = []
            for i in range(chunks):
                for j in range(32):
                    index = i + j
                    val = vals[index]
                    for x in range(3):
                        chunk.append(val[x])
                frames.append(chunk)
                chunk = []
        dfs[k] = np.matrix(frames)
        sizes.append(dfs[k].shape[0])
        
    matrices = []
    for k, v in dfs.items():
        matrices.append(v)
    data = np.concatenate(matrices)

    target = []
    n = 0
    for s in sizes:
        for i in range(s):
            target.append(n)
        n += 1
        
    return data, target
    
# plots the distortion for x k-means so a user can use the elbow method
def elbow_test(data, max_k):
    K = range(1, max_k + 1)
    distortions = []
    for i in range(max_k):
        i += 1
        print(i)
        distortion = cluster(data, targets, 50).Kmeans_test(k = i)
        distortions.append(distortion)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# finds the max k by trying as 1 to max_k k values
# and finding what made the classifier the most accurate
def find_max_k_brute_force(data, targets, max_k, out='a'):
    accuracy = 0.0
    confusion_matrix = 0
    final_tuning_val = 0
    for i in range(max_k):
        i += 1
        print(i)
        a, cmat = cluster(data, targets, 50).Kmeans(output=out, k=i).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
        if a > accuracy:
            accuracy = a
            confusion_matrix = cmat
            final_tuning_val = i
    print('Final d = {}'.format(96))
    print('Final k = {}'.format(k_const*final_tuning_val))
    print('Final Accuracy: {}'.format(accuracy))
    print('Confusion Matrix: \n')
    print(confusion_matrix)

# A final run using 14, 20, 644 as the k-values
# Also runs k-means being added as a feature vs k-means being removed as a feature
# Finally runs the random forrest without k-means
# all of these runs are used for final comparison in the short writeup
def final_run(d, targets):
    k_vals=[14, 20, 644]
    print("K-Means + Random Forest on original data and K-means data as added feature")
    for i in k_vals:
        #a, cmat = cluster(d, targets, 50).Kmeans(output='r', tune=i, k=k_const).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
        a, cmat = cluster(d, targets, 50).Kmeans(output='a', k=i).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
        print('Final d = {}'.format(96))
        print('Final k = {}'.format(i))
        print('Final Accuracy: {}'.format(a))
        print('Confusion Matrix: \n')
        print(cmat)

    print("\n K-Means + Random Forest only on K-means data")
    a, cmat = cluster(d, targets, 50).Kmeans(output='r', k=658).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
    print('Final d = {}'.format(96))
    print('Final k = {}'.format(658))
    print('Final Accuracy: {}'.format(a))
    print('Confusion Matrix: \n')
    print(cmat)

    print("\n Only Random Forest Classifier: ")
    a, cmat = cluster(d, targets, 50).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
    print('Final d = n/a'.format(96))
    print('Final k = n/a'.format(i))
    print('Final Accuracy: {}'.format(a))
    print('Confusion Matrix: \n')
    print(cmat)

def main():
    d, targets = parse_data()
    # elbow_test(d, 50)  # graphs the distortions for k = 1 - k = 50 and then shows the graph to use the elbow test
    final_run(d, targets)
if __name__ == "__main__":
    main()
