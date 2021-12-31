import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
import numpy.random as rng

#Basic LSTM

#Receives a DF vector, normalizes it, and splits it into a number of size-subvectors. Returns a list of such size-subvectors plus a vector of ahead-vectors with future values.
def windowsize(data, size, ahead):
	lx, ly = [], []
	for i in range(len(data)-size-ahead):
		x=data[i:(i+size), :]
		y=data[i+size+ahead-1, 0].squeeze()
		lx.append(x)
		ly.append(y)
	return np.array(lx), np.array(ly)
	
#Revert normalization
def denormalize(scaler, normal):
	denormal = []
	for x in normal:
		denormal.append(scaler.inverse_transform(x))
	return np.array(denormal)

def tts(data,ratio):
	split = []
	for x in data:
		n = int(len(x)*(1.0-ratio))
		split.append(x[:n])
		split.append(x[n:])
	return tuple(split)

#LSTM predictor
def lstmpred(data, inpt):
	exdata = data[data.ticker == inpt['tick']]
	npdata = np.array(exdata[inpt['features']])
	
	#Labels (dates)
	labels = list(map(str, exdata['datetime'].values))
	
	#Scale data
	scaler=mms(feature_range=(-1,1))
	npdata=scaler.fit_transform(npdata)

	#Split data in smaller windows
	nx, ny = windowsize(npdata, inpt['size'], inpt['ahead'])

	#Splitting into train and test sets
	ntrnx, ntstx, ntrny, ntsty = tts([nx, ny], inpt['ratio'])
	
	#Build model
	model = Sequential()
	for i in range(inpt['layers']):
		model.add(LSTM(inpt['neurons'][i], dropout=0.2, return_sequences = True, input_shape = (ntrnx.shape[1], len(inpt['features']))))
	model.add(LSTM(1, dropout=0.2, return_sequences= False))

	#Compile and fit model
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.fit(ntrnx, ntrny, validation_data=(ntstx, ntsty), batch_size=inpt['batch'], epochs=inpt['epochs'])

	#Test
	npreds = model.predict(ntstx)
	preds = scaler.inverse_transform(npreds)
	tsty = scaler.inverse_transform(ntsty.reshape(-1,1))

	plt.figure(figsize=(20,10))
	xaxis = range(len(preds))
	plt.plot(xaxis,preds.squeeze(), color="red")
	plt.plot(xaxis,tsty.squeeze(), color="blue")
	plt.show()
	return model

#Execute another script from python shell
#exec(open('mybvp.py').read())
	
#Read CSV file
data = pd.read_csv('b3_stocks_1994_2020.csv')

#Get all tickers
tickers = ["ABEV3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BPAC11", "BRAP4", "BRDT3", "BRFS3", "BRKM5", "BRML3", "BTOW3", "CCRO3", "CIEL3", "CMIG4", "COGN3", "CRFB3", "CSAN3", "CSNA3", "CVCB3", "CYRE3", "ECOR3", "EGIE3", "ELET3", "ELET6", "EMBR3", "ENBR3", "EQTL3", "FLRY3", "GGBR4", "GNDI3", "GOAU4", "GOLL4", "HAPV3", "HGTX3", "HYPE3", "IGTA3", "IRBR3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LAME4", "LREN3", "MGLU3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RAIL3", "RENT3", "SANB11", "SBSP3", "SMLS3", "SULA11", "SUZB3", "TAEE11", "TIMP3", "TOTS3", "UGPA3", "USIM5", "VALE3", "VIVT4", "VVAR3", "WEGE3", "YDUQ3"]

#Input for LSTM
inpt = {
	'features' : ['close'],
	'target'   : ['close'],
	'size'     : 10,
	'ahead'    : 1,
	'tick'     : "ITUB4",   #Itaú!
	'ratio'    : 0.20,      #Test size
	'layers'   : 3,
	'neurons'  : [150, 50, 50],  #For each layer
	'epochs'   : 100,
	'batch'    : 128  
}

model=lstmpred(data,inpt)
