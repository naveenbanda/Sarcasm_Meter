# Sarcasm-Meter
Sarcasm Detection in Text 

# Preprocessing

Removed reply tweets, tweets having links/non-ascii characters, remaining length<3. 
Saved diveded pos and neg DB as .npy files: 
- posproc.npy
- negproc.py

# Models

1. Pilot Model: Naive Bayes with TFIDF feature vectors.

- BernaoulliNB
- MultinomialNB

2. Final Model: Utilized Linear Dirichlet Allocation (LDA) for topic modeling and other basic features like bigrams for the classification.

