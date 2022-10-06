from json import load
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

import pandas as pd
from sklearn.model_selection import train_test_split

# Remove stopwords and remove words with 2 or less characters using gensim
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
model = keras.models.load_model('mymodel.h5')

df = pd.read_csv("news.csv")

list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)


x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

total_words = len(list(set(list_of_words)))
total_words

tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)



def fake_news_det(news):    
    inp = []
    inp.insert(0,news)

    processed_inp = preprocess(inp[0])
    joined_inp = ' '.join(processed_inp)
    list_inp = []
    list_inp.append(joined_inp)

    inp_seq = tokenizer.texts_to_sequences(list_inp)

    pad_inp = pad_sequences(inp_seq,maxlen = 30, truncating = 'post') 

    op = model.predict(pad_inp)

#     print(op)

# if the predicted value is >0.5 it is real else it is fake
    prediction1 = []
    if op > 0.5:
        prediction1 = "REAL"
    else:
        prediction1 = "FAKE"
    return prediction1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        res0 = fake_news_det(message)
        print(res0)
        return render_template('index.html', res = res0)
    else:
        return render_template('index.html', message=message, prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)