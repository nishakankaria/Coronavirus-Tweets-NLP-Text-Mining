import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
import sklearn.model_selection as model
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import time

start = time.time()

#Read the data from 'Corona_NLP_train.csv' and save it in a dataframe
tweet_data = pd.read_csv('data/Corona_NLP_train.csv', engine='python')

"""1. Compute the possible sentiments that a tweet may have, the second most popular
sentiment in the tweets, and the date with the greatest number of extremely positive tweets.
Next, convert the messages to lower case, replace non-alphabetical characters with whitespaces
and ensure that the words of a message are separated by a single whitespace."""

#Compute the possible sentiments that a tweet may have
print('Q1. Possible sentiments that a tweet may have : ')
#Finding unique sentiment from the column
for sentiment in tweet_data['Sentiment'].unique():
    print(sentiment)
    
print('--------------------------------------------------') 

#The second most popular sentiment in the tweets
print('Q1. Second most popular sentiment in the tweets : ', 
      tweet_data['Sentiment'].value_counts().nlargest(n=2).iloc[[1]].index[0])   

print('--------------------------------------------------')

#Date with the greatest number of extremely positive tweets
dt_filter = tweet_data[tweet_data['Sentiment']=='Extremely Positive'].groupby(['TweetAt'], as_index=False).agg({'Sentiment':'count'})
print('Q1. Date with the greatest number of extremely positive tweets : ', dt_filter.loc[dt_filter['Sentiment'].idxmax()][0])
print('--------------------------------------------------')

#convert all the rows of 'OriginalTweet' column into lower case
tweet_data['OriginalTweet'] = [x.lower() for x in tweet_data['OriginalTweet']]

#drop off those columns which are not required for this coursework
tweet_data.drop(tweet_data.columns[[0, 1, 2, 3]], axis=1, inplace=True)

#This function replaces non-alphabetical characters with whitespaces and ensures that the words of a message are 
#separated by a single whitespace
def replace_non_alphabetic(tweet):  
    tweet = " ".join(re.sub("[^a-zA-Z]", " ", tweet).split())
    return tweet

#Call the function
tweet_data['OriginalTweet'] = tweet_data['OriginalTweet'].apply(replace_non_alphabetic)


""" 2. Tokenize the tweets (i.e. convert each into a list of words), count the total number
of all words (including repetitions), the number of all distinct words and the 10 most frequent
words in the corpus. Remove stop words, words with  2 characters and recalculate the number
of all words (including repetitions) and the 10 most frequent words in the modified corpus. What do you observe? """


#Tokenize the tweets (i.e. convert each into a list of words) and save the result in column 'TokenizeTweet'
tweet_data['TokenizeTweet'] = tweet_data['OriginalTweet'].str.split()

#Count no. of words each row and save the result in 'TokenizeTweet_count' column
tweet_data['TokenizeTweet_count'] = tweet_data['TokenizeTweet'].str.len()

#Count the total number of all words (including repetitions)
print('Q2. Total number of all words (including repetitions) :', tweet_data['TokenizeTweet_count'].sum())
print('--------------------------------------------------')

#Find the number of all distinct words
dist_words = Counter()
tweet_data['TokenizeTweet'].apply(dist_words.update)
print('Q2. Number of all distinct words :', len(dist_words))
print('--------------------------------------------------')

#Find the 10 most frequent words in the corpus
print('Q2. The 10 most frequent words in the corpus :')
for k, v in dist_words.most_common(10):
    print (k)
print('--------------------------------------------------')


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#This function will remove stop words and words with less than or equal to 2 characters
def clean_data(tweet):
    #remove less than 2 characters
    tweet = re.sub(r'\b\w{1,3}\b', ' ', tweet)
    tweet = tweet.split()
    #remove stop_words
    tweet = " ".join([word for word in tweet if word not in stopwords_dict])
    return tweet

stop_words = stopwords.words('english')
#this could have been done without using dictionary, but in order to reduce the runtime of the program and 
#make the process quick, I'm using the Counter here, which has decreased the runtime by 4 seconds.
stopwords_dict = Counter(stop_words)
tweet_data['Modified_TokenizeTweet'] = tweet_data['OriginalTweet'].apply(clean_data)

#Tokenize data for modified corpus
tweet_data['Modified_TokenizeTweet'] = tweet_data['Modified_TokenizeTweet'].str.split()

#Count the total number of all words (including repetitions) in the modified corpus
tweet_data['Modified_TokenizeTweet_count'] = tweet_data['Modified_TokenizeTweet'].str.len()
print("Q2. Total number of all words in the modified corpus (including repetitions) :",tweet_data['Modified_TokenizeTweet_count'].sum())
print('--------------------------------------------------')

#Find the number of all distinct words
dist_words_modified = Counter()
tweet_data['Modified_TokenizeTweet'].apply(dist_words_modified.update)
print("Q2. The number of all distinct words in the modified corpus:",len(dist_words_modified))
print('--------------------------------------------------')

#Find the 10 most frequent words in the corpus
print("Q2. The 10 most frequent words in the modified corpus :")
for k, v in dist_words_modified.most_common(10):
    print (k)
print('--------------------------------------------------')


"""3. Plot a histogram with word frequencies, where the horizontal axis corresponds to
words, while the vertical axis indicates the fraction of documents in a which a word appears.
The words should be sorted in increasing order of their frequencies. Because the size of the data
set is quite big, use a line chart for this, instead of a histogram. In what way this plot can be
useful for deciding the size of the term document matrix? How many terms would you add in a
term-document matrix for this data set?"""

#This function is created to drop/delete duplicate words from each row. 
#This is done so I can calculate the fraction of documents in which a word appears later.
def drop_duplicate_words_row(row):
    words = row
    return ' '.join(np.unique(words).tolist())

# apply the function
tweet_data['TokenizeUniqueWord'] = tweet_data['Modified_TokenizeTweet'].apply(drop_duplicate_words_row)
#tokenized unique words in each row
tweet_data['TokenizeUniqueWord'] = tweet_data['TokenizeUniqueWord'].str.split()

#Count the frequency of each word in documents
unique_words = Counter()
tweet_data['TokenizeUniqueWord'].apply(unique_words.update)

#Plot a line chart with word frequencies, where the horizontal axis corresponds to
#words, while the vertical axis indicates the fraction of documents in a which a word appears.

df_plot = pd.DataFrame(columns=['word','fraction_doc'])
df_plot['word'] = unique_words.keys()
df_plot['fraction_doc'] = unique_words.values()
df_plot['fraction_doc'] = df_plot['fraction_doc']/len(tweet_data)
df_plot = df_plot.sort_values('fraction_doc', ascending=True)


word = np.arange(0, len(df_plot))
fraction_doc = df_plot['fraction_doc']
fig,axs = plt.subplots(1,figsize=(12,9))
plt.plot(word, fraction_doc)  
plt.show()



""" 4. Produce a Multinomial Naive Bayes classier for the Coronavirus Tweets NLP data set using scikit-learn. For this, store
the corpus in a numpy array, produce a sparse representation of the term-document matrix with
a CountVectorizer and build the model using this term-document matrix. What is the error rate
of the classifer? You may want to check the scikit-learn documentation for performing this task. """

#Create a new dataframe consisting of two columns- 'tweet' and 'sentiment'
data = pd.DataFrame(columns=['tweet','sentiment'])
data['tweet'] = tweet_data['OriginalTweet']
data['sentiment'] = tweet_data['Sentiment']

#Reference taken from this site: https://www.kaggle.com/shahraizanwar/covid19-tweets-sentiment-prediction-rnn-85-acc

#This function will clean data of tweet column by removing urls, html tags, digits, mention @, less than 3 characters, stop words
def data_cleaner(tweet):
    
    # remove urls
    tweet = re.sub(r'http\S+', ' ', tweet)
    
    # remove html tags
    tweet = re.sub(r'<.*?>',' ', tweet)
    
    # remove digits
    tweet = re.sub(r'\d+',' ', tweet)
    
    # remove hashtags
    tweet = re.sub(r'#\w+',' ', tweet)
    
    # remove mentions
    tweet = re.sub(r'@\w+',' ', tweet)
    
    #remove less than 2 characters
    tweet = re.sub(r'\b\w{1,3}\b', ' ', tweet)
    
    #removing stop words
    tweet = tweet.split()
    tweet = " ".join([word for word in tweet if word not in stopwords_dict])
    
    return tweet

data['tweet'] = data['tweet'].apply(data_cleaner)

#store the corpus in a numpy array
X = data['tweet'].to_numpy()
Y = data['sentiment'].to_numpy()

#Split data into training and test dataset
#Here I have used the 'train_test_split' to split the data in 80:20 ratio i.e. 
#80% of the data will be used for training the model while 20% will be used for testing the model that is built out of it.
X_train, X_test, Y_train, Y_test = model.train_test_split(X, Y, test_size=0.2)

#produce a sparse representation of the term-document matrix with a CountVectorizer
vectorizer = CountVectorizer()
X_train_v = vectorizer.fit_transform(X_train)
X_test_v = vectorizer.transform(X_test)

#build the naive bayes model using this term-document matrix.
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_v, Y_train)


#predict the new document from the testing dataset
y_pred = naive_bayes_classifier.predict(X_test_v)

#compute the performance measures
accuracy_score = metrics.accuracy_score(Y_test, y_pred)
print("Accuracy:   %0.3f" % accuracy_score)

print('------------------------------')

print("Confusion Matrix:")
print(metrics.confusion_matrix(Y_test, y_pred))

print('------------------------------')

#find the error rate
error_rate = 1 - accuracy_score
print("Error rate:   %0.3f" %error_rate)


# end time
end = time.time()

# total time taken
print(f"Runtime of the program is {end - start}")