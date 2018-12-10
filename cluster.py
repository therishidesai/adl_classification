import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os

class cluster():
    def _load_data(self, sklearn_load_ds, targets, random):
        data = sklearn_load_ds
        X = pd.DataFrame(data)
        self.X_data = sklearn_load_ds
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, targets, test_size=0.3, random_state = random)
        self.random = random
        
        
    def __init__(self, sklearn_load_ds, targets, random):
        self._load_data(sklearn_load_ds, targets, random)
        
    def classify(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy, confusion_matrix(self.y_test, y_pred)
        #print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))
        #print(confusion_matrix(self.y_test, y_pred))

    def Kmeans(self, output='add', tune=1, k=1):
        n_clusters = k
        #n_clusters = int(len(np.unique(self.y_train)) * tune)
        clf = KMeans(n_clusters = n_clusters, random_state = self.random, n_jobs=2)
        clf.fit(self.X_train)
        y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        if output == 'add':
            self.X_train['km_clust'] = y_labels_train
            self.X_test['km_clust'] = y_labels_test
        elif output == 'replace':
            self.X_train = y_labels_train[:, np.newaxis]
            self.X_test = y_labels_test[:, np.newaxis]
        else:
            raise ValueError('output should be either add or replace')
        return self

#32x3 data setup
def parse_data():
    frames = []
    #files = {"Brush_teeth":[]}
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
    

def main():
    d, targets = parse_data()

    accuracy = 0.0
    confusion_matrix = 0
    final_tuning_val = 0

    for i in range(1000):
        i += 1
        print(i)
        a, cmat = cluster(d, targets, 50).Kmeans(output='add', k=i).classify(model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=50))
        if a > accuracy:
            accuracy = a
            confusion_matrix = cmat
            final_tuning_val = i

    print('Final d = {}'.format(96))
    print('Final k = {}'.format(final_tuning_val))
    print('Final Accuracy: {}'.format(accuracy))
    print('Confusion Matrix: \n')
    print(confusion_matrix)
if __name__ == "__main__":
    main()
