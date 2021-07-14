# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
#from sklearn.grid_search import GridSearchCV
%matplotlib inline


'''import bz2
def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(1 if int(x[9]) == 2 else 0)
        texts.append(x[10:].strip())
    return np.array(labels), texts
train_labels, train_texts = get_labels_and_texts('/kaggle/input/amazonreviews/train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('/kaggle/input/amazonreviews/test.ft.txt.bz2')

#data_train['review'][7]
print(train_labels[4])
print(train_texts[4])

# In[6]:
data={"text":train_texts,'stars':train_labels}
data_train=pd.DataFrame(data)
data1={"text":test_texts,'stars':test_labels}
data_test=pd.DataFrame(data1)
'''
import numpy as np
import pandas as pd
def multiclass_metrics(cnf_matrix):
	cnf_matrix=np.asarray(cnf_matrix)
	FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
	FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
	TP = np.diag(cnf_matrix)
	TN = cnf_matrix.sum() - (FP + FN + TP)
	FP = FP.astype(float)
	FN = FN.astype(float)
	TP = TP.astype(float)
	TN = TN.astype(float)

	TP=np.sum(TP)
	TN=np.sum(TN)
	FP=np.sum(FP)
	FN=np.sum(FN)


	accuracy=(TP+TN)/(TP+FP+FN+TN)
	precision=TP/(TP+FP)
	recalll=TP/(FN+TP)
	F1=2*precision*recalll/(precision+recalll)
	sensiti=TP/(TP+FN)
	specifici=TN/(TN+FP)
	numerator=TP*TN - FP*FN
    
	denominator=np.sqrt((TP+FP)*(FN+TN)*(FP+TN)* (TP+FN))
	MCc=numerator/denominator
	G_mean1=np.sqrt(sensiti*precision)
	G_mean2=np.sqrt(sensiti*specifici)
	print('precision:' ,TP/(TP+FP))
	print('recall:',TP/(FN+TP))
	print("F1:",F1)
	print("Specificity:",TN/(TN+FP))
	print("Sensitivity ",TP/(TP+FN))
	print('G-mean1:',np.sqrt(sensiti*precision))
	print("G-mean2",np.sqrt(sensiti*specifici))
	print("MCC :",MCc)
	acc=[]
	pre=[]
	recall=[]
	f1=[]
	specificity=[]
	sensitivity=[]
	GMean1=[]
	Gmean2=[]
	MCC=[]
	tp=[]
	fp=[]
	fn=[]
	tn=[]
	acc.append(accuracy)
	pre.append(precision)
	recall.append(recalll)
	f1.append(F1)
	specificity.append(specifici)
	sensitivity.append(sensiti)
	GMean1.append(G_mean1)
	Gmean2.append(G_mean2)
	MCC.append(MCc)
	tp.append(TP)
	fp.append(FP)
	tn.append(TN)
	fn.append(FN)
	data={'accuracy_all':acc,"precision":pre,'recall':recall,'F1_score':f1,'specificity':specificity,'sensitivity':sensitivity,'Gmean1':GMean1,"Gmean2":Gmean2,"MCC":MCC,"TP":tp,"FP":fp,"TN":tn,"FN":fn,}
	metric=pd.DataFrame(data)
	return metric

#cnf_matrix=[[1025,0,0,20,0,0,0,0,17],[0,0,0,2,0,0,0,0,3],[83,0,63,5,0,0,0,0,0],[18,0,0,330,0,0,0,0,1],[16,0,0,0,165,0,0,0,0],[51,0,0,0,0,0,0,0,0],[2,0,0,1,0,0,0,0,2],[8,0,0,0,0,0,0,0,0],[32,0,0,2,0,0,0,0,154]]


data_train=pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')

data_test=pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
print("hi")

data_train=data_train[:10000]
data_test=data_test[10000:]
# In[7]:



data_train.rename(columns={'text':'title','airline_sentiment':'tag'},inplace=True)
data_test.rename(columns={'text':'title','airline_sentiment':'tag'},inplace=True)
# In[94]:
print('jij')

# In[8]:


# In[88]:


data_train['title']=data_train['title'].astype(str)
data_test['title']=data_test['title'].astype(str)
#data_train
print('fdd')


'''def make_tags(x):   #converting the ratings column into 0's and 1's.  for binary classifier to take place
    if(x<=3):
        return 0
    else:
        return 1
  


# In[10]:


data_train['tag']=data_train['tag'].apply(lambda x: make_tags(x))
data_test['tag']=data_test['tag'].apply(lambda x: make_tags(x))
print('sddsd')
'''
x_train=data_train['title']
y_train=data_train['tag']

test_cnn_data=data_test['title']
#y_test=data_test['tag']

print('sdfsdfsdf')
'''def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''
print('ddsd')
vocab = CountVectorizer().fit(x_train)
print("dwerty")
print(len(vocab.vocabulary_))
#print(x_train[2000])
'''r0 = x[2000]
print(r0)
vocab0 = vocab.transform([r0])
print(vocab0)
"""
    Now the words in the review number 78 have been converted into a vector.
    The data that we can see is the transformed words.
    If we now get the feature's name - we can get the word back!
"""
print("Getting the words back:")
print(vocab.get_feature_names()[19648])
print(vocab.get_feature_names()[10643])
'''

x_train = vocab.transform(x_train)
test_cnn_data=vocab.transform(test_cnn_data)
print("Shape of the sparse matrix: ", x_train.shape)
print(y_train)

#########MULTIONOMIAL NAIVEBAYES
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
print("hih")
model.fit(x_train,y_train.values)
#predmnb = mnb.predict(x_test)
#print("Confusion Matrix for Multinomial Naive Bayes:")
#print(confusion_matrix(y_test,predmnb))
#print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
#print("Classification Report:",classification_report(y_test,predmnb))




pred=model.predict(test_cnn_data)
#print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
'''for i in range(len(y_test)):
    if(y_test[i][0]<0.5):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
'''      
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()

# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#as its a fake news classifier , so identifying a fake class will be a TP
def check_metric(output_class_pred,original_ans):
    rightly_predicted=0
    TP=0
    for i in range(len(y_test)):
        if(original_ans[i]==output_class_pred[i]):
            rightly_predicted+=1
        
        
    print("Overall_acuracy:",rightly_predicted/len(output_class_pred))
    print('TP',TP)
    accuracy=rightly_predicted/len(y_test)
    print(classification_report(original_ans,output_class_pred))
    print(confusion_matrix(original_ans,output_class_pred))
    TN=confusion_matrix(original_ans,output_class_pred)[0][0]
    TP=confusion_matrix(original_ans,output_class_pred)[1][1]
    FP=confusion_matrix(original_ans,output_class_pred)[0][1]
    FN=confusion_matrix(original_ans,output_class_pred)[1][0]
    
    precision=TP/(TP+FP)
    recalll=TP/(FN+TP)
    F1=2*precision*recalll/(precision+recalll)
    sensiti=TP/(TP+FN)
    specifici=TN/(TN+FP)
    numerator=TP*TN - FP*FN
    
    denominator=np.sqrt((TP+FP)*(FN+TN)*(FP+TN)* (TP+FN))
    MCc=numerator/denominator
    G_mean1=np.sqrt(sensiti*precision)
    G_mean2=np.sqrt(sensiti*specifici)
    print('precision:' ,TP/(TP+FP))
    print('recall:',TP/(FN+TP))
    print("F1:",F1)
    print("Specificity:",TN/(TN+FP))
    print("Sensitivity ",TP/(TP+FN))
    print('G-mean1:',np.sqrt(sensiti*precision))
    print("G-mean2",np.sqrt(sensiti*specifici))
    print("MCC :",MCc)
    acc=[]
    pre=[]
    recall=[]
    f1=[]
    specificity=[]
    sensitivity=[]
    GMean1=[]
    Gmean2=[]
    MCC=[]
    tp=[]
    fp=[]
    fn=[]
    tn=[]
    acc.append(accuracy)
    pre.append(precision)
    recall.append(recalll)
    f1.append(F1)
    specificity.append(specifici)
    sensitivity.append(sensiti)
    GMean1.append(G_mean1)
    Gmean2.append(G_mean2)
    MCC.append(MCc)
    tp.append(TP)
    fp.append(FP)
    tn.append(TN)
    fn.append(FN)
    data={'accuracy_all':acc,"precision":pre,'recall':recall,'F1_score':f1,'specificity':specificity,'sensitivity':sensitivity,'Gmean1':GMean1,"Gmean2":Gmean2,"MCC":MCC,"TP":tp,"FP":fp,"TN":tn,"FN":fn}
    metric=pd.DataFrame(data)
    return metric
    
  


        
cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='w', index = False, header=resi.columns,columns=resi.columns)


# In[ ]:



########RANDOMFOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train.values)

pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()

cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)



####DECISION TREE
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
model.fit(x_train,y_train.values)

pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()

cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)




#####SVC
from sklearn.svm import SVC
model = SVC(random_state=101)
model.fit(x_train,y_train.values)
pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()

cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)



####GRADIENT BOOSTING CLASSIFIER
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1,max_depth=5,max_features=0.5,random_state=999999)
model.fit(x_train,y_train.values)

pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()

cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)


    
#####KNN 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train,y_train.values)
pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()
cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)




####XGBOOST CLASSIFIER
import xgboost
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train,y_train)
pred=model.predict(test_cnn_data)
print(y_test)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
output_class_pred=y_test
original_ans=data_test['tag']
original_ans=original_ans.tolist()
cnf_matrix=confusion_matrix(original_ans,output_class_pred)
    

resi=multiclass_metrics(cnf_matrix)
resi.to_csv('results1.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)



print(output_class_pred)
print(original_ans)