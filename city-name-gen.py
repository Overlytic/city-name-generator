#LSTM neural network to generate city names.
#Long term Short Term Memory networks

from __future__ import absolute_import, division, print_function	#compatibility python2.7and python3.5

import os #operating independent functionality... for file exist check

from six import moves #helps pull data from the internet. six is a nice gateway library between python2 and 3
import ssl			  #connect to the internet

import tflearn		  
from tflearn.data_utils import *

#Step 1 - Retrieve the data

path = "US_cities.txt"

if not os.path.isfile(path):
	context = ssl._create_unverified_context()	#not used...unsure about the context thing
	#get dataset
	moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", path) #context=context) works now??

#city name max len
maxlen=20

#vectorise text file --> take all the words and find commonalities.

X, Y, char_idx = textfile_to_semi_redundant_sequences(path, 
	seq_maxlen=maxlen, redun_step=3)

#Create LSTM

g = tflearn.input_data(shape=[None,maxlen,len(char_idx)])
g = tflearn.lstm(g,512,return_seq=True)
g = tflearn.dropout(g,0.5)	#randomly turns off nodes while training. 0.5 -> coefficient ??
g = tflearn.lstm(g,512)	#second layer
g = tflearn.dropout(g,0.5)
g = tflearn.fully_connected(g, len(char_idx), activation= 'softmax') #last layer. softmax... logistic regression type -> good for classification

g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.001)

#Generate cities

m = tflearn.SequenceGenerator(g, dictionary=char_idx, 
								seq_maxlen=maxlen,
								clip_gradients=5.0,
								checkpoint_path='model_us_cities')

#training 

for i in range(40):
	seed = random_sequence_from_textfile(path, maxlen)
	m.fit(X,Y,validation_set=0.1, batch_size=128,
		n_epoch=1, run_id='us cities')
	print("TESTING")
	print(m.generate(30,temperature=1.2,seq_seed=seed))
	print("TESTING")
	print(m.generate(30,temperature=1.0,seq_seed=seed))
	print("TESTING")
	print(m.generate(30,temperature=0.5,seq_seed=seed))