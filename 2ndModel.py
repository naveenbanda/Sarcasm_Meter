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
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from textblob import TextBlob
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

def get_sentiment_score(x):
	scores=[]
	for tweet in x:
		temp=tweet.split()
		l=len(temp)
		first=temp[0:int(l/2)]
		second=temp[1+int(l/2):]
		blob = TextBlob(' '.join(first))
		s1=blob.sentiment.polarity
		blob=TextBlob(' '.join(second))
		s2=blob.sentiment.polarity
		score=0
		if s1<0 and s2>=0:
			score=s1-s2
		elif s1>=0 and s2<0:
			score=s2-s1
		elif s1<=0 and s2<=0:
			score=-1*(s1+s2)
		else:
			score=s1+s2
		scores.append(score)
	
	return np.array(scores).reshape(-1, 1)

'''
def get_topic(x):
    l=len(x)
    dictionary = corpora.Dictionary([i.split() for i in x])
    doc_term_matrix = [dictionary.doc2bow(doc.split()) for doc in x]
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(doc_term_matrix, num_topics=l, id2word = dictionary, passes=50)
    topics=ldamodel.get_topics()
    topics=topics[:,0]
    return (topics)
    #print (ldamodel.print_topics(num_topics=100, num_words=1))    
'''


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

#get_topic(dataset)

#Classify using Naive Bayes: 
from sklearn.naive_bayes import BernoulliNB
#vec, clf = TfidfVectorizer(min_df=5), BernoulliNB()
#td_matrix = vec.fit_transform(dataset)

clf = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vec', TfidfVectorizer(min_df=5,ngram_range=(1,2))),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False)),
        ])),
        ('sentiment', Pipeline([
            ('senti', FunctionTransformer(get_sentiment_score, validate=False)),
        ]))
    
        #('topics', Pipeline([('topic', FunctionTransformer(get_topic, validate=False)),]))
        
    ])),
    ('clf', LinearSVC())])


print ("Length of dataset = "+str(len(dataset)))
print ("Length of the labels = "+str(len(labels)))
X_train, X_test, y_train, y_test = train_test_split(dataset, labels,test_size=0.2, random_state=0)

'''
rfe = RFE(clf, 500)
fit = rfe.fit(X_train, y_train)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_
'''


clf.fit(X_train, y_train)
y_out = clf.predict(X_test)

print("Accuracy on held-out data: ",      str(100*accuracy_score(y_out, y_test))[0:5], "%\n")

from sklearn.metrics import precision_score, recall_score, f1_score
print (precision_score(y_out, y_test))
print (recall_score(y_out, y_test))
print (f1_score(y_out, y_test))
 
#Accuracy on held-out data: MultinomialNB 83.79 %, BernoulliNB 84.49%, DecisionTree=84.40%, RandomForest=82.39%
# After removing stopwords and lemmatizing.. accuracy changed to 84.89% in BernoulliNB. 
# After adding length of the tweet as a feature, nothing changed.
# Accuracy after using unigrams and bigrams both is 85.35%
