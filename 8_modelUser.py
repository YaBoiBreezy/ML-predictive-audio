#Predictive audio generator
#Alexander Breeze
#Using wav files, uncompressed, PCM format
#Similar to wavenet, (rnn to iterate over stream, predict next value), but with mel spectrograms instead of raw audio

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
from tensorflow.keras.callbacks import EarlyStopping
import optuna

#globals
fs = 22050  # Sample rate
ch = 1 #number of channels

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
 librosa.display.specshow(data, sr=fs, hop_length=512, x_axis='time', y_axis='mel')
 plt.colorbar(format='%+2.0f dB')
 plt.title('Mel Spectrogram')
 plt.tight_layout()
 plt.show()

class dataGenerator(keras.utils.Sequence):
 def __init__(self, fileName, chunkSize, batchSize):
  test = readWavFile(fileName)
  hop_length = int(fs / 10)  # Hop length for 10 windows per second
  win_length = hop_length * 2  # Window length to achieve 50% overlap

  # Create mel spectrogram with specified hop length, win length, and FFT window size
  self.melData = librosa.feature.melspectrogram(y=test, sr=fs, hop_length=hop_length, win_length=win_length, n_fft=win_length)

  self.maxer = np.max(self.melData)
  self.logMelData = librosa.power_to_db(self.melData, ref=self.maxer)
  self.dbMax=np.max(self.logMelData)
  self.dbMin=np.min(self.logMelData)
  self.normalizedData = self.logMelData / self.dbMin
  self.data = np.transpose(self.normalizedData)
  self.size = self.data.shape[0]
  self.numChunks = self.size - (chunkSize + 1 + 1)             #<============pred moved right by 1
  self.chunkSize = chunkSize #how long the input to the network is
  self.batchSize = batchSize
  self.indices = np.random.permutation(self.numChunks)

 def __len__(self):
  return int(self.numChunks / self.batchSize)

 def __getitem__(self, i):
  data_points = self.data[self.indices[i:i+self.batchSize, None] + np.arange(self.chunkSize)]
  labels = self.data[self.indices[i:i+self.batchSize] + self.chunkSize + 1]             #<============pred moved right by 1
  return data_points, labels

def makeModel(chunkSize, lstmSize, learningRate):
 model = keras.Sequential()
 model.add(keras.layers.Input(shape=(chunkSize,128)))
 model.add(keras.layers.LSTM(lstmSize))
 model.add(keras.layers.Dense(128, activation='sigmoid'))
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss='MSE')
 #model.compile(optimizer=keras.optimizers.Adam(), loss='MSE')
 #print(model.summary())
 return model



if __name__ == "__main__":
 chunkSize=16
 batchSize=64
 train = dataGenerator("train.wav", chunkSize, batchSize)
 val = dataGenerator("validate.wav", chunkSize, batchSize)
 model = makeModel(chunkSize,380,0.0125)
 early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
 model.fit(train, validation_data=val, epochs=15, callbacks=[early_stopping], verbose=1)

 test = readWavFile("test.wav") [:22050*5] #<<======================================================
 hop_length = int(fs / 10)  # Hop length for 10 windows per second
 win_length = hop_length * 2  # Window length to achieve 50% overlap
 melData = librosa.feature.melspectrogram(y=test, sr=fs, hop_length=hop_length, win_length=win_length, n_fft=win_length)
 maxer = np.max(melData)
 logMelData = librosa.power_to_db(melData, ref=maxer)
 dbMin=np.min(logMelData)
 normalizedData = logMelData / dbMin
 data = np.transpose(normalizedData)

 data1=data
 for i in range(16, len(data1)-2, 2):
  data1[i+1] = data1[i]
 data1 = data1 * dbMin
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data1), ref=maxer))
 #playWav(rawData)

 data2=data
 for i in range(17, len(data2)-2, 2):
  predict=model.predict(np.array([data2[i-17:i-1]]), verbose=0)
  data2[i+1]=predict
 data2 = data2 * dbMin
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data2), ref=maxer))
 #playWav(rawData)

 data3=data
 test3=test
 predict=np.zeros((len(data3),128))
 for i in range(17, len(data3)):
  predict[i]=model.predict(np.array([data2[i-17:i-1]]), verbose=0)
 predict = predict * dbMin
 print(predict.shape)
 #2205 long segments, 5*10 of them, start chopping after 2 seconds
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(predict), ref=maxer), sr=fs, hop_length=hop_length, win_length=win_length, n_fft=win_length)
 print(rawData.shape)
 cut1=random.randint(20,49)
 cut2 = random.randint(20, 49)
 while abs(cut2 - cut1) <= 2:
    cut2 = random.randint(20, 49)
 test3[cut1*2205:(cut1+1)*2205]=0
 test3[cut2*2205:(cut2+1)*2205]=0
 playWav(test3)
 test3[cut1*2205:(cut1+1)*2205]=rawData[cut1*2205:(cut1+1)*2205]
 test3[cut2*2205:(cut2+1)*2205]=rawData[cut2*2205:(cut2+1)*2205]
 playWav(test3)
 exit()

 

 for _ in range(43*2):
  predict=model.predict(np.array([data[-17:-1]]), verbose=0)
  l2Norm = np.linalg.norm(predict)
  print(data[-3])
  data=np.concatenate((data,predict), axis=0)
 data = data * dbMin
 print(data)
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data), ref=maxer))
 print(rawData)
 print(data.shape)
 print(rawData.shape)
 playWav(rawData)
 display(data)
