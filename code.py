#Import all the required packages 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

#load the dataset
data = pd.read_csv("email_spam.csv",encoding='latin-1')
data

#exploratory Data Analysis

data.columns

data.info()

data.shape

data.describe()

#Remove duplicates
data = data.drop_duplicates()

#Find null values
data.isnull().sum()

x = data['v2'].values
y = data['v1'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test= cv.transform(x_test)



svc = SVC()
svc.fit(x_train,y_train)


y_pred = svc.predict(x_test)
y_pred


accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

