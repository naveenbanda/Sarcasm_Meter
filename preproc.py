'''This functions cleans all the tweets.
It first removes all the #tags, then make sure the tweets
does not contain http links, non ASCII charaters or that the
first letter of the tweet is @ (to ensure that the tweet is not out of context).
Then it removes any @tagging and any mention of the word sarcasm or sarcastic.
If after this the tweet is not empty and contains at least 3 words, it is added to the list.
After that all the words in the corpus are lemmatized and then stopwords are removed.
Finally, duplicate tweets are removed.'''

import numpy as np
import csv
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
#from __future__ import unicode_literals
lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
#stemmer=SnowballStemmer('english')

def preprocessing(csv_file_object):
    data=[]
    length=[]
    remove_hashtags = re.compile(r'#\w+\s?')
    remove_friendtag = re.compile(r'@\w+\s?')
    remove_sarcasm = re.compile(re.escape('sarcasm'),re.IGNORECASE)
    remove_sarcastic = re.compile(re.escape('sarcastic'),re.IGNORECASE)   


    for row in csv_file_object:
        if len(row[0:])==1:
            temp=row[0:][0]
            temp=remove_hashtags.sub('',temp)
            if len(temp)>0 and 'http' not in temp and temp[0]!='@' and '\\u' not in temp and temp[:2]!='RT': 
                temp=remove_friendtag.sub('',temp) 
                temp=remove_sarcasm.sub('',temp)
                temp=remove_sarcastic.sub('',temp)
                temp=[lemmatizer.lemmatize(word) for word in temp.split()] #Lemmatize the words
                temp=[word for word in temp if word not in stop_words] #Remove the Stopwords
                #temp=[stemmer.stem(word) for word in temp] #Stemming the words
                temp=' '.join(temp) #remove useless space
                if len(temp.split())>2:
                    data.append(temp)
                    length.append(len(temp.split()))
    data=list(set(data)) #remove duplicate tweets
    data = np.array(data)
    
    return data, length

print ('Extracting data')

### POSITIVE DATA ####
csv_file_object_pos = csv.reader(open('twitDB_sarcasm.csv', 'rU'),delimiter='\n')
pos_data, length_pos = preprocessing(csv_file_object_pos)
#print pos_data

### NEGATIVE DATA ####
csv_file_object_neg = csv.reader(open('twitDB_regular.csv', 'rU'),delimiter='\n')
neg_data, length_neg = preprocessing(csv_file_object_neg)

print ('Number of  sarcastic tweets :', len(pos_data))
print ('Average length of sarcastic tweets :', np.mean(length_pos))
print ('Number of  non-sarcastic tweets :', len(neg_data))
print ('Average length of non-sarcastic tweets :', np.mean(length_neg))

#save the tweets as binary .npy files
np.save('posproc',pos_data)
np.save('negproc',neg_data)
