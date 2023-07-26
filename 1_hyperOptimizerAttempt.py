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
from dtw import dtw
import keras_tuner as kt

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
  self.data = np.reshape(temp, [self.numChunks, chunkSize]) #break it up into chunks

 def __len__(self):
  print(self.numChunks - chunksPS)
  return self.numChunks - chunksPS #subtract 1 second worth of chunks for label offset

 def __getitem__(self, index):
  x = self.data[index:index+chunksPS] #data is 1 second worth of chunks
  y = self.data[index+chunksPS] #label is next chunk
  print("giving data")
  return x, y

'''
def getData(fileName):
  temp=readWavFile(fileName)
  data = np.reshape(temp, [int(temp.shape[0]/fs), fs]) #break it up into 1-second chunks
  x = data[:-1]
  x = np.reshape(x, [x.shape[0], chunks, int(fs/chunks)])
  x = np.concatenate((x, np.zeros((x.shape[0],5,int(fs/chunks)))), axis=1)
  y = data[1:, :int(fs/10)]
  #y = np.reshape(y, [y.shape[0], int(chunks/10), int(fs/chunks)])
  return x, y
'''

def makeModel(hp):
 hp_dnnSize = hp.Int('dnnSize', min_value=2, max_value=1024, step=2, sampling='log') #exponential steps, 2 4 8 16 32 64 128...
 hp_LSTMSize = hp.Int('LSTMSize', min_value=512, max_value=4096, step=512)

 model = keras.Sequential()
 model.add(keras.layers.TimeDistributed(keras.layers.Dense(hp_dnnSize, activation='tanh'))) #dnn to reduce every chunk to important features like cnn
 model.add(keras.layers.LSTM(hp_LSTMSize, input_shape=(chunksPS, chunkSize), activation='tanh', recurrent_activation='tanh'))
   #model.add(keras.layers.Lambda(lambda x: x[:, -1:, :])) #Select last 5 chunks from output
 model.add(keras.layers.Dense(chunkSize, activation='tanh')) #dnn to resize output chunk to 882
   #model.add(keras.layers.Flatten())

 model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='MAE')
 print(model.summary())
 return model




 



def main():
 train=dataGenerator("train.wav")
 val=dataGenerator("validate.wav")
 test=dataGenerator("test.wav")



 h1=readWavFile("h1.wav")
 h2=readWavFile("h2.wav")
 h3=readWavFile("h3.wav")


 tuner = kt.Hyperband(makeModel, objective='val_loss', max_epochs=10, factor=3)
 tuner.search(train, validation_data=val, epochs=50)
 best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
 #print(f"""The hyperparameter search is complete. {best_hps.get('dnnSize')}  {best_hps.get('lstmSize')}.""")

 
 #    training_performance = model.fit(trainingSignGenerator, validation_data=validationSignGenerator, epochs=25,verbose=2)
 #    test_performance = model.evaluate(testSignGenerator)
 #    #print(test_performance)

#    return model, training_performance.history['loss'][-1], training_performance.history['val_loss'][-1]
main()

