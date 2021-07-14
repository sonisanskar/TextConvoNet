# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import bz2
import pickle
import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import json
#get_ipython().run_line_magic('matplotlib', 'inline')
'''
from tensorflow.keras.optimizers import Adam

# In[3]:
trainfile = bz2.BZ2File('../input/amazonreviews/train.ft.txt.bz2','r')
lines = trainfile.readlines()

sent_analysis = []
def sent_list(docs,splitStr='__label__'):
    for i in range(1,len(docs)):
        text=str(lines[i])
        splitText=text.split(splitStr)
        #print(i)
        secHalf=splitText[1]
        text=secHalf[2:len(secHalf)-1]
        sentiment=secHalf[0]
        sent_analysis.append([text,sentiment])
    return sent_analysis

sentiment_list=sent_list(lines[:1000000],splitStr='__label__')

train_df = pd.DataFrame(sentiment_list,columns=['Text','Sentiment'])

data_train=train_df[:4000]
data_test=train_df[4000:5000]
'''
#a=input('path of the taining dataset with fields as title and tag(0,1) ')
#b=input('path of test dataset')
#data_train=pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv')


# In[4]:


#data_train


# In[5]:

data_train=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data_test=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

data_train=data_train[:40000]
data_test=data_test[40000:]



# In[8]:
data_train.rename(columns={'review':'title','sentiment':'tag'},inplace=True)
data_test.rename(columns={'review':'title','sentiment':'tag'},inplace=True)

#data_train['rating'].value_counts()
#print('training_dataset',data_train)
#print('training_dataset',data_test)

# In[9]:

#print(data_train)

def make_tags(x):   #converting the ratings column into 0's and 1's.  for binary classifier to take place
    if(x=="negative"):
        return 0
    else:
        return 1
  


# In[10]:


data_train['tag']=data_train['tag'].apply(lambda x: make_tags(x))
data_test['tag']=data_test['tag'].apply(lambda x: make_tags(x))

#print(data_train)

count0=(data_train['tag']==0).sum()
count1=(data_train['tag']==1).sum()
if(count0>count1):
    imbalance_ratio=(count0)/count1
else:
    imbalance_ratio=(count1)/count0
# In[11]:

print('imbalance_ratio',imbalance_ratio)
#print(data_train)


# In[12]:



def no_of_words_in_paragraph(x):
    return len(list(x))

data_train['no_of_words_in_paragraph']=data_train['title'].apply(lambda x:no_of_words_in_paragraph(x))

data_test['no_of_words_in_paragraph']=data_test['title'].apply(lambda x:no_of_words_in_paragraph(x))



print(data_train)
avg=data_train['no_of_words_in_paragraph'].mean()
maxim=data_train['no_of_words_in_paragraph'].max()
print('average paragraph length',data_train['no_of_words_in_paragraph'].mean())
print('maximum para length',data_train['no_of_words_in_paragraph'].max())
print('hii')
excess=(data_train['no_of_words_in_paragraph']>avg).sum()
excess_ratio=excess/len(data_train)
print('excess_ratio',excess_ratio)


#applying sentence tokenizer
import nltk.data 
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle') 
# Loading PunktSentenceTokenizer using English pickle file 
def make_sent_token(x):
    return tokenizer.tokenize(x) 
#converting each paragraph into separate sentences


# In[13]:


data_train['sentence_token']=data_train['title'].apply(lambda x: make_sent_token(x))

data_test['sentence_token']=data_test['title'].apply(lambda x: make_sent_token(x))


# In[15]:


#data_train.drop(columns=['uniqueID','date','usefulCount','condition','drugName'],inplace=True,axis=1)# dropping irrelevant columns


# In[16]:


#data_test.drop(columns=['uniqueID','date','usefulCount','condition','drugName'],inplace=True,axis=1)


# In[17]:


#data_train


# In[18]:


data_train['no_of_sentences']=data_train['sentence_token'].apply(lambda x:len(x))


# In[19]:


data_test['no_of_sentences']=data_test['sentence_token'].apply(lambda x:len(x))


# In[20]:
avg_sen_length=data_train['no_of_words_in_paragraph'].sum()/data_train['no_of_sentences'].sum()
print(avg_sen_length)

#max(data_train['no_of_sentences'])##no of rows in sentence matrix which is to be feed in model(max number of sentence in any paragraph)


# In[21]:


#len(data_train[data_train['no_of_sentences']==92]['review'])


# In[22]:


#max(data_test['no_of_sentences'])


# In[23]:


def max_length_of_sentence(x,y):
    sen=x
    nu=y
    #print(sen)
    ma=0
    if(nu>1):
        l=sen.split('.')
        #print(l)
        for i in range(len(l)):
            k=l[i].replace(',','')
            maxi=len(k.split())
            #print(maxi)
            if(maxi>ma):
                ma=maxi
        return ma
    else:
        return len(sen.split())
        
    


# In[24]:


data_train['max_words_in_sentence']=data_train.apply(lambda x: max_length_of_sentence(x.title,x.no_of_sentences),axis=1)


# In[25]:


data_test['max_words_in_sentence']=data_test.apply(lambda x: max_length_of_sentence(x.title,x.no_of_sentences),axis=1)


# In[26]:


#max(data_train['max_words_in_sentence'])## number of columns in the data to be feeded


# In[27]:

x1=max(data_train['no_of_sentences'])
y1=max(data_train['max_words_in_sentence'])

x2=max(data_test['no_of_sentences'])
y2=max(data_test['max_words_in_sentence'])

if(x1>=x2):
    m=x1
    print(m)
    m=m
else:
    m=x2
    m=m
    
if(y1>=y2):
    n=y1
else:
    n=y2

#So each para will be converted to a m*n matrix
if(m<5):
    m=6
else:
    m+=2
print('x1,x2,y1,y2',x1,x2,y1,y2)

print("m-->",m,n)
#So each para will be converted to a m*n matrix


# In[28]:




# # Major part starts here ..... Now converting the paragraph into required matrix

# In[29]:


import re
import string 
from nltk import word_tokenize
from nltk.corpus import stopwords
def make_tokens(text):     ##Converting into single tokens in order to create the vocabulary
    return word_tokenize(text)


data_train['tokens']=data_train['title'].apply(lambda x: make_tokens(x))
data_test['tokens']=data_test['title'].apply(lambda x: make_tokens(x))


# In[30]:


#data_train['tokens']


# In[ ]:


#from gensim import models
#word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
#word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)



embeddings_index = {}
f = open('../input/glove6b300dtxt/glove.6B.300d.txt')
for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
    embeddings_index[word] = coefs
f.close()

print('GloVe data loaded')

# In[ ]:


all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
para_max=max(training_sentence_lengths)

vocab=len(TRAINING_VOCAB)

# In[ ]:


#len(TRAINING_VOCAB)


# In[ ]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), char_level=False)
tokenizer.fit_on_texts(data_train['title'])       # we assigned values 


# In[ ]:


train_word_index = tokenizer.word_index


# In[ ]:


#print(train_word_index)


# In[ ]:


#data_train.to_csv('medic_train.csv')
#data_test.to_csv('medic_test.csv')


# In[ ]:


def make_train_seq(x):
    return tokenizer.texts_to_sequences(x)
data_train['train_seq']=data_train['sentence_token'].apply(lambda x:make_train_seq(x) )
data_test['train_seq']=data_test['sentence_token'].apply(lambda x:make_train_seq(x) )


# In[ ]:


#(data_train['train_seq'])   # here every para has been encoded


# In[ ]:
#print(data_train)




# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
def padding(x):    #now padding each sentence to a length of n...number of columns
    MAX_SENTENCE_LENGTH=n  #(no of columns)
    return pad_sequences(x,maxlen=MAX_SENTENCE_LENGTH,padding='post')

data_train['padded']=data_train['train_seq'].apply(lambda x:padding(x))
data_test['padded']=data_test['train_seq'].apply(lambda x:padding(x))


# In[ ]:


#(data_train.padded[8])


# In[ ]:



## More code adapted from the keras reference (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
# prepare embedding matrix 
from tensorflow.keras.layers import Embedding
from tensorflow.keras.initializers import Constant

## EMBEDDING_DIM =  ## seems to need to match the embeddings_index dimension
EMBEDDING_DIM = embeddings_index.get('a').shape[0]
print(EMBEDDING_DIM)
#num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
 #= np.zeros(len(train_word_index) + 1, EMBEDDING_DIM)
train_embedding_weights = np.zeros((len(train_word_index)+1, 
 EMBEDDING_DIM))
for word, i in train_word_index.items():
    #print("sd")
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        train_embedding_weights[i] = embedding_vector
print(train_embedding_weights.shape)
        # words not found in embedding index will be all-zeros.
        

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
#embedding_layer = Embedding(num_words,
                          #  EMBEDDING_DIM,
                          #  embeddings_initializer=Constant(embedding_matrix),
                          #  input_length=MAX_SEQUENCE_LENGTH,
                          #  trainable=False)


#EMBEDDING_DIM=300
#train_embedding_weights = np.zeros((len(train_word_index)+1, 
 #EMBEDDING_DIM))
#for word,index in train_word_index.items():
 #train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
#print(train_embedding_weights.shape)


# In[43]:


def make_full_para(x):     #92 cross 192 matrix of a paragraph.   (m*n)
    l=len(x)
    h=m-l    #no. of extra rows to be added
    z=[0]*h*n       #1D vector(#addding extra lines for zeroes as padding)
    z=np.reshape(z,(h,n))    #reshaping it to match the dimension of paragraph
    s=x.tolist()+z.tolist()
    return s 


# In[ ]:





# In[ ]:


data_train['full_para']=data_train['padded'].apply(lambda x : make_full_para(x))
data_test['full_para']=data_test['padded'].apply(lambda x : make_full_para(x))


# In[ ]:


#data_train.full_para


# In[ ]:


def create_1d_para(x):
    l=[]
    for i in x:
        l+=i    #concatenating all the sentences in a para into a single 1 d arrray
    return l
        
    


# In[ ]:

data_train['single_d_array']=data_train['full_para'].apply(lambda x: create_1d_para(x) )
data_test['single_d_array']=data_test['full_para'].apply(lambda x: create_1d_para(x) )


# In[ ]:


#train_cnn_data=np.array(data_train['single_d_array'].tolist())


# In[ ]:


train_cnn_data=np.array(data_train['single_d_array'].tolist())
test_cnn_data=np.array(data_test['single_d_array'].tolist())


# In[ ]:

from sklearn.model_selection import train_test_split
y_train=data_train['tag'].values



# In[ ]:

print('Startting the training')
#from __future__ import print_function
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,GRU,LSTM,SpatialDropout1D,Reshape
from tensorflow.keras.layers import Embedding,concatenate
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D,MaxPool2D,MaxPool3D,GlobalAveragePooling2D,Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# In[ ]:

filter_sizes = [1,2,3,4]
num_filters = 32
embed_size=300
embedding_matrix=train_embedding_weights
max_features=len(train_word_index)+1
maxlen=m*n

def get_model():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((m, n, 300))(x)
    #print(x)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 2), 
                                                                                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 3),
                                                                                    activation='relu')(x)
    
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 4),
                                                                                    activation='relu')(x)
    
    
    
    
    
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 1), 
                                                                                    activation='relu')(x)
    conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 2), activation='relu')(x)
    
    conv_6 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 3),
                                                                                    activation='relu')(x)
    
    
    
    maxpool_0 = MaxPool2D()(conv_0)
    maxpool_0=Flatten()(maxpool_0)
    maxpool_1 = MaxPool2D()(conv_1)
    maxpool_1=Flatten()(maxpool_1)
    maxpool_2 = MaxPool2D()(conv_2)
    maxpool_2 = Flatten()(maxpool_2)
    
    maxpool_4 = MaxPool2D()(conv_4)
    maxpool_4=Flatten()(maxpool_4)
    maxpool_5 = MaxPool2D()(conv_5)
    maxpool_5=Flatten()(maxpool_5)
    maxpool_6 = MaxPool2D()(conv_6)
    maxpool_6=Flatten()(maxpool_6)
    #maxpool_7 = MaxPool2D()(conv_7)
   # maxpool_7=Flatten()(maxpool_7)
    z = concatenate([maxpool_0, maxpool_1,maxpool_2],axis=1)
    w=concatenate([maxpool_4, maxpool_5,maxpool_6],axis=1)    
    #w=concatenate([maxpool_4, maxpool_5,maxpool_6],axis=1)    
    #z = concatenate([maxpool_0, maxpool_1,maxpool_2,maxpool_4, maxpool_5,maxpool_6],axis=1)
    #z = concatenate([maxpool_0, maxpool_1,maxpool_4, maxpool_5],axis=1)
    
    #z = Flatten()(z)
    z=concatenate([w,z],axis=1)
    z=Dense(units=64,activation="relu")(z)
    z = Dropout(0.4)(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


# In[ ]:


model=get_model()


# In[ ]:


print(model.summary())


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()
history=model.fit(train_cnn_data, y_train,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )
end = datetime.datetime.now()
diff1= (end - start)
print('time taken by text_6',diff1)






pred=model.predict(test_cnn_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
for i in range(len(y_test)):
    if(y_test[i][0]<0.5):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()

# In[ ]:

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#as its a fake news classifier , so identifying a fake class will be a TP
def check_metric(output_class_pred,original_ans,diff1):
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
    data={'accuracy_all':acc,"precision":pre,'recall':recall,'F1_score':f1,'specificity':specificity,'sensitivity':sensitivity,'Gmean1':GMean1,"Gmean2":Gmean2,"MCC":MCC,"TP":tp,"FP":fp,"TN":tn,"FN":fn,"traintime":diff1,"Exceeding_ratio":excess_ratio,"imbalance_ratio":imbalance_ratio,"Average_length_of_paragraph":avg,"Maximum_length_of_a_paragraph":maxim,"Average_length_of_sentences":avg_sen_length,"Maximum_length_of_a_sentence_in_a_paragraph":n,"Maximum_no_of_sentence_in_any_paragraph":m,"Vocabular_size":vocab,"label0":count0,"label1":count1}
    metric=pd.DataFrame(data)
    return metric

print(history.history.keys())
    
resi=check_metric(output_class_pred,original_ans,diff1)
resi.to_csv('results_text.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)



#####

filter_sizes = [1,2,3,4]
num_filters = 32
embed_size=300
embedding_matrix=train_embedding_weights
max_features=len(train_word_index)+1
maxlen=m*n
def get_model():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((m, n, 300))(x)
    #print(x)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 2), 
                                                                                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 3),
                                                                                    activation='relu')(x)
    
    #conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 4),
                                                                                    #activation='relu')(x)
    
    
    
    
    
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 1), 
                                                                                    activation='relu')(x)
    conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 2), activation='relu')(x)
    
    #conv_6 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 3),
                                                                                    #activation='relu')(x)
    
    
    
    maxpool_0 = MaxPool2D()(conv_0)
    maxpool_0=Flatten()(maxpool_0)
    maxpool_1 = MaxPool2D()(conv_1)
    maxpool_1=Flatten()(maxpool_1)
    #maxpool_2 = MaxPool2D()(conv_2)
    #maxpool_2 = Flatten()(maxpool_2)
    
    maxpool_4 = MaxPool2D()(conv_4)
    maxpool_4=Flatten()(maxpool_4)
    maxpool_5 = MaxPool2D()(conv_5)
    maxpool_5=Flatten()(maxpool_5)
    #maxpool_6 = MaxPool2D()(conv_6)
    #maxpool_6=Flatten()(maxpool_6)
    #maxpool_7 = MaxPool2D()(conv_7)
   # maxpool_7=Flatten()(maxpool_7)
        
    #w=concatenate([maxpool_4, maxpool_5,maxpool_6],axis=1)    
    #z = concatenate([maxpool_0, maxpool_1,maxpool_2,maxpool_4, maxpool_5,maxpool_6],axis=1)
    #z = concatenate([maxpool_0, maxpool_1,maxpool_4, maxpool_5],axis=1)
    w=concatenate([maxpool_4, maxpool_5],axis=1)    
    #z = concatenate([maxpool_0, maxpool_1,maxpool_2,maxpool_4, maxpool_5,maxpool_6],axis=1)
    z = concatenate([maxpool_0, maxpool_1],axis=1)
    
    #z = Flatten()(z)
    z=concatenate([w,z],axis=1)
    #z = Flatten()(z)
    #z=concatenate([w,z],axis=1)
    z=Dense(units=64,activation="relu")(z)
    z = Dropout(0.4)(z)
        
    outp = Dense(1, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

    return model


# In[ ]:


model=get_model()


# In[ ]:


print(model.summary())


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()

history=model.fit(train_cnn_data, y_train,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )

end = datetime.datetime.now()
diff1= (end - start)
print('time taken by text_4',diff1)





pred=model.predict(test_cnn_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
for i in range(len(y_test)):
    if(y_test[i][0]<0.5):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()

# In[ ]:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#as its a fake news classifier , so identifying a fake class will be a TP


resi=check_metric(output_class_pred,original_ans,diff1)
resi.to_csv('results_text.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)







# In[ ]:





# In[ ]:


## now perparing training data for yoon kim model


# In[ ]:


def create_single_line_para(x):
    l=[]
    for i in x:
        l+=i    #concatenating all the sentences in a para into a single 1 d arrray
    return l
        


# In[ ]:


data_train['create_single_line_para']=data_train['train_seq'].apply(lambda x: create_single_line_para(x) )
data_test['create_single_line_para']=data_test['train_seq'].apply(lambda x: create_single_line_para(x) )


# In[ ]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
yoon_kim_train_data=np.array(data_train['create_single_line_para'].tolist())
yoon_kim_train_data=pad_sequences(yoon_kim_train_data,maxlen=para_max,padding='post')

# In[ ]:
yoon_kim_test_data=np.array(data_test['create_single_line_para'].tolist())
yoon_kim_test_data=pad_sequences(yoon_kim_test_data,maxlen=para_max,padding='post')


#from __future__ import print_function
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,GRU,LSTM
from tensorflow.keras.layers import Embedding,concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# In[ ]:


train_y=pd.get_dummies(y_train)


# In[ ]:


trains_y=train_y[[0,1]].values


# In[ ]:


embed_size=300
embedding_matrix=train_embedding_weights
max_features=len(train_word_index)+1
maxlen=para_max 
max_sequence_length=para_max
MAX_SEQUENCE_LENGTH=para_max
EMBEDDING_DIM=300


#model3 yoon kim


# In[ ]:


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, trainable=True, extra_conv=False):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=2)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    #conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    #pool = MaxPooling1D(pool_size=2)(conv)

    #if extra_conv==True:
        #x = Dropout(0.01)(l_merge)  
    #else:
        # Original Yoon Kim model
        #x = Dropout(0.001)(pool)
    x = Flatten()(l_merge)
    
    x = Dropout(0.5)(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) 
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(2, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['acc'])
    model.summary()
    return model


# In[ ]:


model1 = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                 True)


# In[ ]:


training_data=yoon_kim_train_data


# In[ ]:


testing_data=yoon_kim_test_data


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()

hist = model1.fit(training_data, trains_y,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )
end = datetime.datetime.now()
diff1= (end - start)
print('time taken by yoon',diff1)


# In[ ]:


pred=model1.predict(testing_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
#output_class_pred=[]
for i in range(len(y_test)):
    m=max(y_test[i])
    if(y_test[i].index(m)==0):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()


# In[ ]:


#as its a fake news classifier , so identifying a fake class will be a TP
resi=check_metric(output_class_pred,original_ans,diff1)

resi.to_csv('results_text', mode='a', index = False, header=resi.columns,columns=resi.columns)



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Dropout, Embedding, concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, ZeroPadding1D
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
from tensorflow.keras.layers import Concatenate, Dot, Multiply, RepeatVector
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Lambda, Permute

#from tensorflow.keras.layers.core import Reshape, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
#from tensorflow.keras.constraints import maxnorm
#from tensorflow.keras.regularizers import l2

def ConvNet_vdcnn(embeddings, max_sequence_length, num_words, embedding_dim, trainable=True, extra_conv=False):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    
    
# 4 pairs of convolution blocks followed by pooling
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sequences)
    
    
# 4 pairs of convolution blocks followed by pooling
    for filter_size in [64, 128, 256, 512]:
    
    # each iteration is a convolution block
        for cb_i in [0,1]:
            conv=(Conv1D(filter_size, 3, padding="same",activation='relu'))(conv)
        #model_1.add(BatchNormalization())
        #model_1.add(Activation("relu"))
            conv=(Conv1D(filter_size, 1, padding="same",activation='relu'))(conv)
        #model_1.add(BatchNormalization())
        #model_1.add(Activation("relu"))
    
        conv=(MaxPooling1D(pool_size=2, strides=3))(conv)

# model.add(KMaxPooling(k=2))
    conv=(Flatten())(conv)
    conv=(Dense(4096, activation="relu"))(conv)
    conv=(Dense(2048, activation="relu"))(conv)
    conv=(Dense(2048, activation="relu"))(conv)
#(Dense(9, activation="softmax"))

    preds = Dense(2, activation='softmax')(conv)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',metrics=['acc'])
    print(model.summary())
    return model
    
    
    
model1 = ConvNet_vdcnn(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                 True)




# In[ ]:


training_data=yoon_kim_train_data


# In[ ]:


testing_data=yoon_kim_test_data


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()

hist = model1.fit(training_data, trains_y,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )
end = datetime.datetime.now()
diff1= (end - start)
print('time taken by yoon',diff1)


# In[ ]:


pred=model1.predict(testing_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
#output_class_pred=[]
for i in range(len(y_test)):
    m=max(y_test[i])
    if(y_test[i].index(m)==0):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()


# In[ ]:


#as its a fake news classifier , so identifying a fake class will be a TP
resi=check_metric(output_class_pred,original_ans,diff1)

resi.to_csv('results_text', mode='a', index = False, header=resi.columns,columns=resi.columns)





def ConvNet_clstm(embeddings, max_sequence_length, num_words, embedding_dim, trainable=True, extra_conv=False):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    convs = []
    filter_sizes = [10, 20, 30, 40]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='valid', activation='relu')(embedded_sequences)
        convs.append(l_conv)

    cnn_feature_maps = Concatenate(axis=1)(convs)
    sentence_encoder = LSTM(64,return_sequences=False)(cnn_feature_maps)
    fc_layer =Dense(128, activation="relu")(sentence_encoder)
    #output_layer = Dense(9,activation="softmax")(fc_layer)

    #model_1 = Model(inputs=[text_input_layer], outputs=[output_layer])
    preds = Dense(2, activation='softmax')(fc_layer)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['acc'])
    model.summary()
    return model


model1 = ConvNet_clstm(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                 True)




# In[ ]:


training_data=yoon_kim_train_data


# In[ ]:


testing_data=yoon_kim_test_data


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()

hist = model1.fit(training_data, trains_y,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )
end = datetime.datetime.now()
diff1= (end - start)
print('time taken by yoon',diff1)


# In[ ]:


pred=model1.predict(testing_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
#output_class_pred=[]
for i in range(len(y_test)):
    m=max(y_test[i])
    if(y_test[i].index(m)==0):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()


# In[ ]:


#as its a fake news classifier , so identifying a fake class will be a TP
resi=check_metric(output_class_pred,original_ans,diff1)

resi.to_csv('results_text', mode='a', index = False, header=resi.columns,columns=resi.columns)




def ConvNet_lstm(embeddings, max_sequence_length, num_words, embedding_dim, trainable=True, extra_conv=False):
    
    
    embedding_layer = Embedding(num_words,embedding_dim,weights=[embeddings],input_length=max_sequence_length,trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    sentence_encoder = LSTM(64,return_sequences=False)(embedded_sequences)
    fc_layer =Dense(128, activation="relu")(sentence_encoder)
    #output_layer = Dense(9,activation="softmax")(fc_layer)
    #model_1 = Model(inputs=[text_input_layer], outputs=[output_layer])
    preds = Dense(2, activation='softmax')(fc_layer)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])
    model.summary()
    return model

    
   
  


model1 = ConvNet_lstm(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                 True)




# In[ ]:


training_data=yoon_kim_train_data


# In[ ]:


testing_data=yoon_kim_test_data


# In[ ]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

import time, datetime
start = datetime.datetime.now()

hist = model1.fit(training_data, trains_y,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )
end = datetime.datetime.now()
diff1= (end - start)
print('time taken by yoon',diff1)


# In[ ]:


pred=model1.predict(testing_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
#output_class_pred=[]
for i in range(len(y_test)):
    m=max(y_test[i])
    if(y_test[i].index(m)==0):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()


# In[ ]:


#as its a fake news classifier , so identifying a fake class will be a TP
resi=check_metric(output_class_pred,original_ans,diff1)

resi.to_csv('results_text', mode='a', index = False, header=resi.columns,columns=resi.columns)



#resi.to_csv('results.csv', mode='a', index = False, header=resi.columns,columns=resi.columns)

