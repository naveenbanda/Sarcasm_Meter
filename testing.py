""" This function loads the trained model and we can give text as input and
get the classification result """

import nltk
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import pickle
import feature_extract
import topic
import heapq

vec_file = open("vecdict.p", "rb")
vec = pickle.load(vec_file)
vec_file.close()

classifier_file = open("classif.p", "rb")
classifier = pickle.load(classifier_file)
classifier_file.close()

topic_mod_file = open("topic_mod.p", "rb")
topic_mod = pickle.load(topic_mod_file)
topic_mod_file.close()

#BASIC TEST

while (True):
    print ("Enter the text to test: ")
    text = input()
    features = []
    features.append(feature_extract.dialogue_act_features(text, topic_mod))
    features = np.array(features)
    feature_vec = vec.transform(features)
    ans = classifier.predict(feature_vec)
    if ans==1:
        print ("The text is sarcastic")
    else:
        print ("The text is non-sarcastic")

    print ("The score of the text is: "+str(classifier.decision_function(feature_vec)))

