import socket
import sys
import time
import threading
import warnings
import numpy as np 
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model

df= pd.read_csv('data/train.csv')

# warnings.filterwarnings('ignore')
max_features=20000
tokenizer= Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(df['comment_text']))

def tokenize(random_tweet):
	tokenized_comment=tokenizer.texts_to_sequences(random_tweet)
	tokenized_comment=pad_sequences(tokenized_comment,200)
	return tokenized_comment

def load_model(m='model.json',w='weights.h5'):
	json_file=open(m,'r')
	model=json_file.read()
	json_file.close()
	model=model_from_json(model)
	model.load_weights(w)
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	return model

model = load_model()

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)

if len(sys.argv)!=3:
	print("script_name IP_address port_no")
	exit()

host_name=str(sys.argv[1])
port=int(sys.argv[2])


#binding
server.bind((host_name,port))
print("hostname and port bound together")
print("server is now listening")

server.listen(10)#we will only accept 10 connections
client_list=[]

def clientJoins(client_socket,ip):
	welcome="Welcome to this chatroom"
	welcome=welcome.encode()
	client_socket.send(welcome)
	# print(client_list)
	while(1):
		try:
			# print("client has joined try")
			msg=client_socket.recv(2048)
			if msg:
				msg=msg.decode()
				response="<"+ip[0]+">: "+msg

				msg= tokenize(msg)
				print(model.predict(msg))

				print(response)
				broadcast(response,client_socket)
			else:
				remove(client_socket)
		except:
			continue

def broadcast(msg,client_socket):
	# print("broadcast is called")
	for client in client_list:
		# print("For loop executed")
		if client!=client_socket:
			try:
				# print("try is executed")
				msg=msg.encode()
				# print("broadcasting to ")
				# print(client)
				client.send(msg)
			except:
				# print("except is executed")
				client.close()
				remove(client)


def remove(client_socket):
	if client_socket in client_list:
		client_socket.remove(client_socket)

i=1
while(1):
	client_socket,ip=server.accept()#connection is socket of client and ip is IP address of client
	client_list.append(client_socket)
	print(ip[0]," is connected")

	threading.Thread(target=clientJoins,args=(client_socket,ip)).start()

client_socket.close()
server.close()

				



# print(ip,"is now connected and ONLINE")
# while(1):
# 	response=input(str(">>"))
# 	response=response.encode()
# 	client_socket.send(response)
# 	print("response sent")
# 	request=client_socket.recv(1024)
# 	request=request.decode()
# 	print("Client: ",request)