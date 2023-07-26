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

#globals
fs = 44100  # Sample rate
ch = 1 #number of channels
chunksPS=50 #number of chunks per second
chunkSize=int(fs/chunksPS) #number of values per chunk (882)

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
  temp=readWavFile(fileName)
  self.numChunks=int(temp.shape[0]/chunkSize)
  self.numSeconds=int(temp.shape[0]/fs)
  self.data = np.reshape(temp, [self.numChunks, chunkSize]) #break it up into chunks

 def __len__(self):
  #return 1
  return chunksPS #one batch for each chunk offset within a second, 50 chunks/sec so 50 batches each with offset i

 def __getitem__(self, index):
  i=index
  x = np.reshape(self.data[i:i+((self.numSeconds-1)*chunksPS)],[self.numSeconds-1, chunksPS, chunkSize])
  y = np.reshape(self.data[i+chunksPS::chunksPS],[self.numSeconds-1, chunkSize])
  return x, y


def makeModel(dnnSize, lstmSize, learningRate):
 model = keras.Sequential()
 model.add(keras.layers.Dense(dnnSize, activation='tanh', input_shape=(chunksPS,chunkSize))) #dnn to reduce every chunk to important features like cnn
 model.add(keras.layers.LSTM(lstmSize, input_shape=(chunksPS, chunkSize), activation='tanh', recurrent_activation='tanh'))
 model.add(keras.layers.Dense(chunkSize, activation='tanh')) #dnn to resize output chunk to 882
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate),loss='MAE')
 print(model.summary())
 return model



def main():
 train=dataGenerator("train.wav")
 x,y=train.__getitem__(4)
 print(x.shape)
 print(y.shape)
 val=dataGenerator("validate.wav")
 test=readWavFile("test.wav")
 
 model=makeModel(256, 128, 0.001)
 model.fit(train, validation_data=val, epochs=1)

 test=np.reshape(test, [1500,882])
 for i in range(1500):
  inp=np.array([test[-50:]])
  predict=model.predict(inp, verbose=0)
  print(predict)
  test=np.concatenate((test,predict), axis=0)
 test=np.reshape(test,[3000*882])
 playWav(test)
 exit()
 

 bestLoss=100
 bestParams=[]
 for dnnSize in [2**i for i in range (5,9)]:
  for lstmSize in [2**i for i in range (7,11)]:
   for learningRate in (1e-3,1e-5,1e-7):
    model=makeModel(dnnSize, lstmSize, learningRate)
    t = model.fit(train, validation_data=val, epochs=3)
    best=min(t.history['val_loss'])
    if best<bestLoss:
     bestLoss=best
     bestParams=[dnnSize,lstmSize,learningRate]
 print(bestParams)
 print(bestLoss)
 
 #[256, 128, 0.001]

 #    test_performance = model.evaluate(testSignGenerator)
 #    #print(test_performance)

#    return model, training_performance.history['loss'][-1], training_performance.history['val_loss'][-1]
main()

import librosa
import librosa.display
mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
