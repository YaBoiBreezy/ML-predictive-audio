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

#globals
fs = 44100  # Sample rate
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

class dataGenerator(keras.utils.Sequence):
 def __init__(self, fileName):
  self.rawData = readWavFile(fileName)
  self.melData = librosa.feature.melspectrogram(y=self.rawData, sr=fs) #create mel spectrogram, left all args as default
  self.data=np.transpose(self.melData)
  self.size = self.data.shape[0] #number of timesteps. 25840 = 512 values per timestep, ~86 timesteps/second
  self.dim = self.data.shape[1] #128

 def __len__(self):
  #return 1
  return self.size - nwtp - 1

 def __getitem__(self, i):
  x = self.data[i:i+nwtp]
  y = self.data[i+nwtp]
  return np.array([x]), np.array([y])


def makeModel(cnnSize, lstmSize, learningRate):
 model = keras.Sequential()
 model.add(keras.layers.Input(shape=(nwtp,128)))
 model.add(keras.layers.LSTM(lstmSize,activation='tanh',recurrent_activation='tanh'))
 model.add(keras.layers.Dense(128, activation='tanh'))
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate),loss='MSE')
 print(model.summary())
 return model


def main():
 print("Started")
 train=dataGenerator("train.wav")
 val=dataGenerator("validate.wav")
 test=np.transpose(librosa.feature.melspectrogram(y=readWavFile("test.wav"), sr=fs))
 print("Loaded data")

 test=readWavFile("test.wav")
 test=test[:int(test.shape[0]/6)]
 a=librosa.griffinlim(librosa.stft(test))
 b=librosa.feature.inverse.mel_to_audio(librosa.feature.melspectrogram(y=test,sr=fs), sr=fs)
 c=librosa.feature.inverse.mfcc_to_audio(librosa.feature.mfcc(y=test,sr=fs), sr=fs)
 playWav(a)
 playWav(b)
 playWav(c)
 exit()


 '''
 bestLoss=100
 bestParams=[]
 count=1
 for cnnSize in [1]:
  for learningRate in (1e-4,1e-5):
   for lstmSize in [2**i for i in range (5,10)]:
    model=makeModel(cnnSize, lstmSize, learningRate)
    print(f"COUNT: {count}")
    print(f"lstm: {lstmSize},  lr: {learningRate}")
    count+=1
    t = model.fit(train, validation_data=val, epochs=5)
    best=min(t.history['val_loss'])
    if best<bestLoss:
     bestLoss=best
     bestParams=[cnnSize,lstmSize,learningRate,np.argmin(t.history['val_loss'])] #[3]=index of best, aka epoch with lowest loss
 print(bestParams)
 print(bestLoss)
 '''


 model=makeModel(0, 128, 0.0001)
 model.fit(train, validation_data=val, epochs=8)

 newLen=86*10  #86 timesteps/second
 for i in range(newLen):
  predict=model.predict(np.array([test[-nwtp:]]), verbose=0)
  test=np.concatenate((test,predict), axis=0)
  if (100*i)%newLen==0:
   print(predict)
   print(str((100*i)/newLen)+"%")
 i=input("finished generating, enter filename to save and play sound!\n")
 rawData=librosa.feature.inverse.mel_to_audio(np.transpose(test)) #uses griffin-lim, so automatically reconstructs phase to go with mel
 print(rawData.shape)
 write(i, fs, rawData)
 playWav(rawData)

main()

#make super overtrained example to verify approach is feasible
#use optimizer, 2x LSTM, variable window size/count