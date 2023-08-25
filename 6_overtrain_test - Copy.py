#Predictive audio generator
#Alexander Breeze
#Using wav files, uncompressed, PCM format
#PCM (Pulse Code Modulation) is series of ints, each int is amplitude of sound wave at timestamp

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import librosa
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, cheby1, ellip

#globals
fs = 22050  # Sample rate
ch = 1 #number of channels
nwtp = 100 #number of windows to make a prediction

def recordWav(time, fileName):
 myrecording = sd.rec(int(time * fs), samplerate=fs, channels=ch)
 sd.wait()  # wait until recording is finished
 write(fileName, fs, myrecording)  # save as WAV file

def playWavFile(fileName):
 data, b = sf.read(fileName, dtype='float32')
 sd.play(data, fs)
 status = sd.wait()  # wait until file is done playing

def readWavFile(fileName):
 data, b = sf.read(fileName, dtype='float32')
 return data

def playWav(rawData):
 sd.play(rawData, fs)
 status = sd.wait()  # wait until file is done playing

def display(data):
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(data, sr=fs, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel Spectrogram')
  plt.tight_layout()
  plt.show()

class dataGenerator(keras.utils.Sequence):
 def __init__(self, fileName):
  test = readWavFile(fileName)
  self.melData = librosa.feature.melspectrogram(y=test, sr=fs) #create mel spectrogram
  self.maxer=np.max(self.melData)
  self.logMelData = librosa.power_to_db(self.melData, ref=self.maxer)
  self.data = np.transpose(self.logMelData)
  self.size = self.data.shape[0] #101 timesteps
  self.dim = self.data.shape[1] #128
  self.indices = np.random.permutation(self.size)           #librosa.db_to_power(log_melData, ref=np.max)

 def __len__(self):
  return self.size
  return self.size - nwtp - 1

 def __getitem__(self, i): #this part definitely works
  if i==0:
   return np.array([self.data[i:i+nwtp]]), np.array([self.data[i+nwtp]])
  if i==1:
   return np.array([self.data[i:i+nwtp]]), np.array([self.data[0]])
  x = np.concatenate((self.data[i:],self.data[:(i+nwtp)%self.size]), axis=0)
  y = self.data[(i+nwtp)%self.size]
  return np.array([x]), np.array([y])


def makeModel(cnnSize, lstmSize, learningRate):
 model = keras.Sequential()
 #model.add(keras.layers.Input(shape=(nwtp,128)))
 model.add(keras.layers.LSTM(lstmSize,activation='tanh',recurrent_activation='tanh', return_sequences=True))
 model.add(keras.layers.LSTM(lstmSize,activation='tanh',recurrent_activation='tanh'))
 model.add(keras.layers.Dense(128, activation='linear'))
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate),loss='MSE')
 #print(model.summary())
 return model


def main():
 print("Started")
 test=dataGenerator("slow.wav")
 maxer=test.maxer
 print("Loaded data")
 
 

 model=makeModel(0, 128, 0.0001)
 model.fit(test, epochs=60)
 
 '''
 predict=model.predict(test.__getitem__(0)[0], verbose=0)
 print(predict)
 print(test.__getitem__(0)[1])
 print(mean_squared_error(predict, test.__getitem__(0)[1]))
 exit()
 '''

 baseLen = 100
 test=test.__getitem__(0)[0][0]
 print(test.shape)
 newLen = baseLen + 86*5  #86 timesteps/second
 for i in range(baseLen, newLen):
  predict=model.predict(np.array([test[-100:]]), verbose=0)
  test=np.concatenate((test,predict), axis=0)
 i=input("finished generating, enter filename to save and play sound!\n")
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(test), ref=maxer)) #uses griffin-lim, so automatically reconstructs phase to go with mel
 print(rawData.shape)
 playWav(rawData)
 display(test)
 write("./resultWavs/"+i, fs, rawData)

main()

#make super overtrained example to verify approach is feasible
#look into mel hyperparameters
#use optimizer, 2x LSTM, variable window count