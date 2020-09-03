import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import matplotlib.pyplot as plt 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

st.title('Toxic Comment Classifier')
st.sidebar.title('Toxic Comment Classifier')
st.markdown('## This dashboards visualizes the training data and predicts toxicity of the comments')
st.markdown('### Toxicity is divided into 6 classes: toxic, severe_toxic, obscene, threat, insult, identity_hate')

# st.sidebar.button('Load training data')
# st.sidebar.button('Load test data')
@st.cache(persist=True)
def load_data(data_URL):
	df=pd.read_csv(data_URL)
	return df



if st.sidebar.checkbox('Load test data'):
	data_URL='data/test.csv'
	df=load_data(data_URL)
else:
	data_URL="data/train.csv"
	df=load_data(data_URL)


st.sidebar.subheader('Explore data')
rows=st.sidebar.slider('Show tabular data ')
if(rows>0):
	st.markdown('**DataFrame till %d rows: **' %rows)
	st.write(df.head(rows))

# st.sidebar.button('Describe data')
if st.sidebar.button('Describe data'):
	st.markdown('**Description:**')
	st.write(df.describe())

# st.sidebar.button('Display Null values count')
if st.sidebar.button('Display Null values count'):
	st.markdown('**Count of NULL values for all columns:**')
	st.write(df[df.isna()==True].count())

st.sidebar.subheader('Visualize training data')

toxic_counts={}
for i in range(6):
    toxic_counts[df.columns[i+2]]=df[df[df.columns[i+2]]==1].count()[0]
    
if st.sidebar.checkbox('include normal comments'):
	normal_comments_count=(df.count())[0]-sum(toxic_counts.values())
	temp_dict={'normal':normal_comments_count}
	chart_data= {**temp_dict,**toxic_counts}
else:
	chart_data=toxic_counts

select= st.sidebar.selectbox('Visualization type',['None','Pie Chart', 'Histogram'],key=2)

if select=='Pie Chart':
	fig_total= go.Figure([go.Pie(labels=list(chart_data.keys()),values=list(chart_data.values()))])
	fig_total.show()
if select=='Histogram':
	plot_val=go.Bar(x=list(chart_data.keys()),y=list(chart_data.values()))
	fig= go.Figure(plot_val)
	fig.show()


st.sidebar.subheader('Generate random tweet')
random_tweet=(df['comment_text'].sample(n=1))
if st.sidebar.checkbox('Generate random tweet',key=5):
	st.markdown('### %s'%random_tweet.iloc[0])

st.sidebar.subheader('Process tweet for prediction')

# @st.cache(persist=True)
def tokenize(random_tweet):
	max_features=20000
	tokenizer= Tokenizer(num_words=max_features)
	tokenizer.fit_on_texts(list(df['comment_text']))

	tokenized_comment=tokenizer.texts_to_sequences(random_tweet)
	tokenized_comment=pad_sequences(tokenized_comment,200)
	return tokenized_comment

if st.sidebar.checkbox('Process'):

	tokenized_comment=tokenize(random_tweet)
	st.write(tokenized_comment)


st.sidebar.subheader('Deep Learning Model')
# @st.cache(persist=True)
def load_model(m='model.json',w='weights.h5'):
	json_file=open(m,'r')
	model=json_file.read()
	json_file.close()
	model=model_from_json(model)
	model.load_weights(w)
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model=load_model()

if st.sidebar.checkbox('Model Summary'):
	st.markdown('### Check you  streamlit terminal for summary')
	st.write(model.summary())

if st.sidebar.button('Predict for random generated tweet'):
	st.markdown('### Here 0 indicates comment being toxic and similarly 5 is prob of comment being identity_hate')
	st.write(model.predict(tokenized_comment))
