import scipy.io as scio
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
dataset = './dataset/MLEXPDatasets.mat'
dict_data = scio.loadmat(dataset)
X = dict_data['train_data'].astype('float32')
min_max_scaler = preprocessing.StandardScaler()
X = min_max_scaler.fit_transform(X)
y = dict_data['train_target'].astype('float32')
y = y.reshape(y.shape[0],)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svc = SVC(kernel='rbf', class_weight='balanced',C=28.22425,gamma=0.0158125)
print(cross_val_score(svc,X, y,cv=5,scoring='accuracy'))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=50)
print(cross_val_score(rfc,X, y,cv=5,scoring='accuracy'))
MLPclassifier = MLPClassifier(hidden_layer_sizes=(350,300),max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=1,  weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                     metric='minkowski', metric_params=None, n_jobs=None)
a=cross_val_score(knn,X, y,cv=5,scoring='f1_micro')
count = sum(a)
print(a)
print(count/5)
