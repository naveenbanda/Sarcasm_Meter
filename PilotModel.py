'''BernoulliNB gave slightly better results than MultinomialNB on just TF-IDF feature vector.'''

import numpy as np

#Load the binary files of sarcastic and non-sarcastic tweets
sarcasm=np.load("posproc.npy")
neutral=np.load("negproc.npy")

#Print sample data
print ("10 sample sarcastic lines:")
print (sarcasm[:10])
print ("10 sample non-sarcastic lines:")
print (neutral[:10])


#Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

labels=[]
sarcasm_size=len(sarcasm)
print ("Total sarcastic lines = "+str(sarcasm_size))
neutral_size=len(neutral)
print ("Total non-sarcastic lines = "+str(neutral_size))

for i in range(0,sarcasm_size):
    labels.append(1)
for i in range(0,neutral_size):
    labels.append(0)
print (len(labels))

dataset=np.concatenate([sarcasm,neutral])
print ("Total length of dataset = "+str(len(dataset)))

#Classify using Naive Bayes: 
from sklearn.naive_bayes import BernoulliNB
vec, clf = TfidfVectorizer(min_df=5), BernoulliNB()
td_matrix = vec.fit_transform(dataset)
#svd = TruncatedSVD(n_components=300, random_state=42)
#svd_matrix=svd.fit_transform(td_matrix)
print ("Shape of matrix = "+str(td_matrix.shape))
print ("Length of the labels = "+str(len(labels)))
X_train, X_test, y_train, y_test = train_test_split(td_matrix, labels,test_size=0.2, random_state=0)

clf.fit(X_train, y_train)
y_out = clf.predict(X_test)

print("Accuracy on held-out data: ",      str(100*accuracy_score(y_out, y_test))[0:5])
 
from sklearn.metrics import precision_score, recall_score, f1_score
print ("Precision: ", str(precision_score(y_out, y_test)))
print ("Recall: ", str(recall_score(y_out, y_test)))
print ("F1-Score: ", str(f1_score(y_out, y_test)))
#Accuracy on held-out data: MultinomialNB 83.79 %, BernoulliNB 84.49%, DecisionTree=84.40%, RandomForest=82.39%
# After removal of stopwords and lemmatization, accuracy 84.52%
