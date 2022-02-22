import pandas as pd
import matplotlib.pyplot as plt
import re
import wordcloud
import nltk
import spacy
from textblob import TextBlob
import streamlit as st
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS


data = pd.read_csv('Elon_musk.csv',encoding='latin1')
afinn = pd.read_csv('Afinn.csv',encoding = 'latin1')


df1 = pd.DataFrame(data['Text'])

def clean_tweet(Text):
    Text = re.sub('@[A-Za-z0–9_a-zA-Z0-9_a-zA-Z]+', ' ', Text)
    Text = re.sub('https?:\/\/\S+', ' ', Text)
    Text = re.sub('[^a-zA-z]',' ',Text)
    return Text
df1.Text = df1.Text.apply(clean_tweet)

afinn = afinn.set_index('word')['value'].to_dict()


nlt = spacy.load('en_core_web_sm')

def sentiment_value(Text :str = None):
    sent_score = 0
    if Text:
        sentence = nlt(Text)
        for word in sentence:
            sent_score += afinn.get(word.lemma_,0)
    return sent_score
df1['sent_value'] = df1['Text'].apply(sentiment_value)

def sentiment(sent_value):
    result = ''
    if sent_value < 0:
        result = 'Negative'
    if sent_value == 0:
        result = 'Neutral'
    if sent_value > 0 :
        result = 'Positive'
    return result
df1['sentiment'] = df1.sent_value.apply(sentiment)
st.header('By using Afinn evaluation Criteria')
st.bar_chart(df1.sentiment.value_count())



#
df2 =  pd.DataFrame(data['Text'])
def clean_tweet(Text):
    Text = re.sub('@[A-Za-z0–9_a-zA-Z0-9_a-zA-Z]+', ' ', Text)
    Text = re.sub('[^a-zA-z]',' ',Text)
    return Text
df2.Text = df2.Text.apply(clean_tweet)

def sub_count(Text):
    return TextBlob(Text).sentiment.subjectivity
def pol_count(Text):
    return TextBlob(Text).sentiment.polarity
df2['subjectivity'] = df2.Text.apply(sub_count)
df2['polarity'] = df2.Text.apply(pol_count)

def sentiment(polarity):
    result = ''
    if polarity < 0:
        result = 'Negative'
    if polarity == 0:
        result = 'Neutral'
    if polarity > 0 :
        result = 'Positive'
    return result
df2['sentiment'] = df2.polarity.apply(sentiment)
st.header('By using TextBlob')
st.bar_chart(df2.sentiment.value_count())

df2 =[Text.strip() for Text in df1.Text]
join_df2=' '.join(df2)

no_punc_text=join_df2.translate(str.maketrans('','',string.punctuation))

# Tokenization
text_tokens = word_tokenize(join_df2)
print(text_tokens[0:50])
len(text_tokens)

# Remove stopword 

my_stop_words = stopwords.words('english')
sw_list = ['\x92','rt','ye','yeah','haha','Yes','U0001F923','I','U','F','C','b']
my_stop_words.extend(sw_list)

no_stop_tokens=[word for word in text_tokens if not word in my_stop_words]
lower_words = [Text.lower() for Text in no_stop_tokens]


ps=PorterStemmer()
stemmed_tokens=[ps.stem(word) for word in lower_words]

# join the data agin for Lemmatization
nlp=spacy.load('en_core_web_sm')
doc=nlp(' '.join(lower_words))

lemmas=[token.lemma_ for token in doc]
clean_tweets=' '.join(lemmas)

# feature extraction 
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(lemmas)

def plot_cloud(wordcloud):
    # set fig size
    plt.figure(figsize = (40,30))
    # display image
    plt.imshow(wordcloud)
    # no axis details
    plt.axis('off');

# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('pron')
stopwords.add('rt')
stopwords.add('yeah')
wordcloud = WordCloud(width = 3000, height = 2000, background_color ='black', max_words = 100,stopwords=stopwords).generate(clean_tweets)
st.write(plot_cloud(wordcloud))    