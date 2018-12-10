import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import os

class cluster():
    def _load_data(self, sklearn_load_ds, targets):
        data = sklearn_load_ds
        X = pd.DataFrame(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, targets, test_size=0.3)
        
        
    def __init__(self, sklearn_load_ds, targets):
        self._load_data(sklearn_load_ds, targets)
        
    def classify(self, model=LogisticRegression()):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print('Accuracy: {}'.format(accuracy_score(self.y_test, y_pred)))


    def Kmeans(self, output='add'):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters = n_clusters, random_state=42)
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
            frames.append(pd.read_table(fname, sep='\s+').values)
        dfs[k] = np.concatenate(frames, axis=0)
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
    #print(d)
    print(d.shape)
    print(len(targets))
    #cluster(d, targets).classify()
    cluster(d, targets).Kmeans(output='replace').classify(model=SVC())
    #cluster(d, targets)
    #print(load_digits())
    print(len(load_digits().data[0]))
    #print(len(load_digits().target))
    print(load_digits().data.shape)
if __name__ == "__main__":
    main()
