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

#globals
fs = 44100  # Sample rate
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

def playWav(data):
 sd.play(data, fs)
 status = sd.wait()  # wait until file is done playing


class dataGenerator(keras.utils.Sequence):
 def __init__(self, fileName):
  self.data = readWavFile(fileName)
  self.size=self.data.shape[0]

 def __len__(self):
  return 1000

 def __getitem__(self, index):
  i=random.randint(0,self.size-1001)
  x = self.data[i:i+1000]
  y = self.data[i+1000]
  return np.array([x]), np.array([y])


def makeModel(dnnSize, lstmSize, learningRate):
 model = keras.Sequential()
 model.add(keras.layers.Dense(dnnSize, activation='tanh', input_shape=(1,1000)))
 model.add(keras.layers.Dense(dnnSize, activation='tanh'))
 model.add(keras.layers.Dense(dnnSize, activation='tanh'))
 model.add(keras.layers.Dense(dnnSize, activation='tanh'))
 model.add(keras.layers.Dense(1, activation='tanh'))
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate),loss='MAE')
 print(model.summary())
 return model


def main():
 train=dataGenerator("train.wav")
 val=dataGenerator("validate.wav")
 test=readWavFile("test.wav")
 
 #model=makeModel(64, 64, 1e-5)
 #model.fit(train, validation_data=val, epochs=100)
 #exit()

 
 '''
 bestLoss=100
 bestParams=[]
 count=1
 for dnnSize in [2**i for i in range (2,6)]:
  for lstmSize in [2**i for i in range (2,6)]:
   for learningRate in (1e-1,1e-2,1e-3,1e-4,1e-5):
    model=makeModel(dnnSize, lstmSize, learningRate)
    print(f"COUNT: {count}")
    count+=1
    t = model.fit(train, validation_data=val, epochs=5)
    best=min(t.history['val_loss'])
    if best<bestLoss:
     bestLoss=best
     bestParams=[dnnSize,lstmSize,learningRate]
 print(bestParams) #[32, 16, 0.0001]
 print(bestLoss)
 exit()
 '''
 
 model=makeModel(300, 512, 0.00001)
 model.fit(train, epochs=100)

 newLen=fs*1
 for i in range(newLen):
  predict=model.predict(np.array([[test[-1000:]]]), verbose=0)
  test=np.concatenate((test,predict[0][0]), axis=0)
  if (100*i)%newLen==0:
   print(predict)
   print(str((100*i)/newLen)+"%")
 i=input("finished generating, enter filename to save and play sound!")
 playWav(test)
 write(i, fs, test)
 exit()




 #    test_performance = model.evaluate(testSignGenerator)
 #    #print(test_performance)

#    return model, training_performance.history['loss'][-1], training_performance.history['val_loss'][-1]
main()
