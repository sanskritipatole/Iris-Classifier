import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load data set
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
db = pd.read_csv(url, names=names)
#summarize the dataset

#1.dimensions of the dataset
'''
We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
'''
#shape
print(db.shape)

#2.peek at the data
#head(no of rows you wanna see)
#print(db.head(5))

#3.statistical summar-- count,mean,max, some percentiles
#description
#print(db.describe())

#4.class distribution
#print(db.groupby('class').size())

#data visualization\
"""
we nw have a basic idea about the data. we need some visualization
2 types:
1.Univariate plots to better understand each attribute
2.Multivariate plots to better understand the relationships between attributes.
"""

#1plot univariate
#db.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

#histogram
#db.hist()
#pyplot.show()

#multivariate
#scatter plot matrix
#scatter_matrix(db)
#pyplot.show()

#evaluate some algorithms
#1.seperate out a validation dataset

#create a validation dataset
#split the datasets into test and validation sets
array = db.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#2.test harness
#WE will use stratifies 10-fold cross validationto estimate model accuracy
#for every 10 sets 9 will be trained on one will be tested
#the order is mixed so that all of them go through the procedure

#lets build models and evaluate our best options

#spot check algorithms
#LR=logistic regression
#LDA=linear discriminant analysis
#CART=classification and regression trees
#NB=gaussian naive bayes
#SVM=support vector machine
models=[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate each model in turn
results=[]
names=[]

for name, model in models:
    kfold=StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results=cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))
#SVM got highest accuracy of 98%

#make pedictions
model= SVC(gamma= 'auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

#evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))