
## STUDENT ID: 5155437
## STUDENT NAME: Zishuo Li


## Question 2

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)       # make sure you run this line for consistency 
x = np.random.uniform(1, 2, 100)
y = 1.2 + 2.9 * x + 1.8 * x**2 + np.random.normal(0, 0.9, 100)
plt.scatter(x,y)
plt.show()

## (c)


def compute_loss(x, y, w, c):
    loss = np.sum(np.sqrt((1.0/c**2)*(y-np.matmul(w, x))**2+1)-1)
    return loss


def compute_gradient(x, y, w, c):
    grad = np.sum(x * (-(y-np.matmul(w, x)) / np.sqrt(c**2*(y-y_pred)**2+c**4)), axis=1)
    return grad


def gradient_descent(x, y, alpha, iter_num=100):
    loss = []
    x_0 = np.ones(x.shape)
    x = np.concatenate((x_0[np.newaxis, :], x[np.newaxis, :]), axis=0)
    W = np.array([1, 1], dtype=np.float64)
    c = 2

    for i in range(iter_num):
        loss.append(compute_loss(x, y, W, c))
        W = W - alpha * compute_gradient(x, y, W, c)

    return loss


if __name__ == '__main__':
    alphas = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7, 10e-8, 10e-9]
    losses = []
    for i in range(len(alphas)):
        losses.append(gradient_descent(x, y, alphas[i]))
        print(losses[i])

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(ax.flat):
        print(i, ax)
        # losses is a list of 9 elements . Each element is an array of length 100 the loss at each iteration for
        # that particular step size
        ax.plot(losses[i])
        print(alphas[i])
        ax.set_title(" step size : {}".format(alphas[i])) # plot titles
    plt.tight_layout() # plot formatting
    plt.show()





## Question 3

# (c)
# YOUR CODE HERE





# Question 5

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def create_dataset():
    X, y = make_classification( n_samples=1250,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2,
                                random_state=5,
                                n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 3 * rng.uniform(size = X.shape)
    linearly_separable = (X, y)
    X = StandardScaler().fit_transform(X)
    return X, y


# (a)
warnings.filterwarnings("ignore")
X, y = create_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifiers = {
    'Decision Trees':DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Logistic Regression': LogisticRegression(),
    'MLP': MLPClassifier(),
    'SVM Classifier': SVC()
}
import os
if not os.path.exists('./pic'):
    os.mkdir('./pic')

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    plt.figure()
    plotter(clf, X, X_test, y_test, name)
    plt.savefig('./pic/%s.png'%name)

def plotter(classifier, X, X_test, y_test, title, ax=None):
    # plot decision boundary for given classifier
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), 
                            np.arange(y_min, y_max, plot_step)) 
    Z = classifier.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax:
        ax.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        ax.set_title(title)
    else:
        plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)
        plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test)
        plt.title(title)


# (b)
train_samples = np.hstack([X_train,y_train.reshape(-1,1)])
acc_dict = {
    'Decision Trees':[],
    'Random Forest': [],
    'AdaBoost':[],
    'Logistic Regression': [],
    'MLP': [],
    'SVM Classifier': []
}


size = 50
for name, clf in classifiers.items():
    acc=[]
    for t in range(10):
        _subset=train_samples[np.random.choice(len(train_samples), size)]
        _X_train, _y_train = _subset[:,:-1], _subset[:,-1]
        clf.fit(_X_train, _y_train)
        acc.append(clf.score(X_test, y_test))
    acc_dict[name].append(np.mean(acc))

for size in range(100,1001,100):
    for name, clf in classifiers.items():
        acc=[]
        for t in range(10):
            _subset=train_samples[np.random.choice(len(train_samples), size)]
            _X_train, _y_train = _subset[:,:-1], _subset[:,-1]
            clf.fit(_X_train, _y_train)
            acc.append(clf.score(X_test, y_test))
        acc_dict[name].append(np.mean(acc))

plt.rcParams['figure.figsize'] = [10, 5]
plt.figure()
train_set_size = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900','1000']
plt.plot(train_set_size,acc_dict['Decision Trees'],'b',label='Decision Trees')
plt.plot(train_set_size,acc_dict['Random Forest'],'orange', label='Random Forest')
plt.plot(train_set_size,acc_dict['AdaBoost'],'g', label='AdaBoost')
plt.plot(train_set_size,acc_dict['Logistic Regression'],'red',label='Logistic Regression')
plt.plot(train_set_size,acc_dict['MLP'],'m',label='MLP')
plt.plot(train_set_size,acc_dict['SVM Classifier'],'y',label='SVM Classifier')
plt.legend(loc=4)
plt.title('Results')
plt.savefig('./Results.png')

# (c)
train_samples = np.hstack([X_train,y_train.reshape(-1,1)])
time_dict = {
    'Decision Trees':[],
    'Random Forest': [],
    'AdaBoost':[],
    'Logistic Regression': [],
    'MLP': [],
    'SVM Classifier': []
}

size = 50
for name, clf in classifiers.items():
    _time=[]
    for t in range(10):
        _subset=train_samples[np.random.choice(len(train_samples), size)]
        _X_train, _y_train = _subset[:,:-1], _subset[:,-1]
        start_time = time.time()
        clf.fit(_X_train, _y_train)
        _time.append(time.time() - start_time)
    time_dict[name].append(np.mean(_time))

for size in range(100,1001,100):
    for name, clf in classifiers.items():
        _time=[]
        for t in range(10):
            _subset=train_samples[np.random.choice(len(train_samples), size)]
            _X_train, _y_train = _subset[:,:-1], _subset[:,-1]
            start_time = time.time()
            clf.fit(_X_train, _y_train)
            _time.append(time.time() - start_time)
        time_dict[name].append(np.mean(_time))

plt.figure()
train_set_size = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900','1000']
plt.plot(train_set_size,time_dict['Decision Trees'],'b',label='Decision Trees')
plt.plot(train_set_size,time_dict['Random Forest'],'orange', label='Random Forest')
plt.plot(train_set_size,time_dict['AdaBoost'],'g', label='AdaBoost')
plt.plot(train_set_size,time_dict['Logistic Regression'],'red',label='Logistic Regression')
plt.plot(train_set_size,time_dict['MLP'],'m',label='MLP')
plt.plot(train_set_size,time_dict['SVM Classifier'],'y',label='SVM Classifier')
plt.legend(loc='best')
plt.title('Training Time')
plt.savefig('./TrainingTime.png')