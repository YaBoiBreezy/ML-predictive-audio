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
  librosa.display.specshow(data, sr=fs, x_axis='time', y_axis='mel')
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
  self.numChunks = self.size - (chunkSize + 1)             #<============pred moved right by 1
  self.chunkSize = chunkSize #how long the input to the network is
  self.batchSize = batchSize
  self.indices = np.random.permutation(self.numChunks)

 def __len__(self):
  return int(self.numChunks / self.batchSize)

 def __getitem__(self, i):
  data_points = self.data[self.indices[i:i+self.batchSize, None] + np.arange(self.chunkSize)]
  labels = self.data[self.indices[i:i+self.batchSize] + self.chunkSize]             #<============pred moved right by 1
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






def objective(trial):
    # Define the search space for hyperparameters
    chunkSize = trial.suggest_int("chunkSize", 1, 20)
    batchSize = trial.suggest_int("batchSize", 16, 128)
    lstmSize = trial.suggest_int("lstmSize", 100, 1000)
    learningRate = trial.suggest_float("learning_rate", low=1e-9, high=1e-1, log=True)
    
    # Create data generators
    train = dataGenerator("train.wav", chunkSize, batchSize)
    val = dataGenerator("validate.wav", chunkSize, batchSize)
    
    # Create and compile model
    model = makeModel(chunkSize, lstmSize, learningRate)

    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train the model
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    history = model.fit(train, validation_data=val, epochs=15, callbacks=[early_stopping], verbose=1)
    
    # Get the validation loss from the final epoch
    val_loss = history.history['val_loss'][-2]
    
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

exit()

####:_#{'chunkSize': 13, 'batchSize': 62, 'lstmSize': 379, 'learning_rate': 0.0124697989313039}. loss: 0.01897384412586689.
#####:#{'chunkSize': 4, 'batchSize': 21, 'lstmSize': 818, 'learning_rate': 0.016351511400111925}. loss: 0.01431801076978445.

def main():
 #test=dataGenerator("test.wav", 100, 64)
 #model=makeModel(100, 1024)
 #model.fit(test, epochs=60)
 '''
 bestLoss=np.inf
 bestParams=[0,0,0,0,0]
 for chunkSize in [50,150]:
  for batchSize in [32,128]:
   for lstmSize in [256,1028]:
    for secondLSTM in [0,256,1028]: #0 means it doesn't get put because it =False
     for secondDNN in [0,256,1028]:
      print(f"Chunk size: {chunkSize}   Batch size: {batchSize}   LSTM size: {lstmSize}    secondLSTM: {secondLSTM}    secondDNN: {secondDNN}")
      train=dataGenerator("train.wav", chunkSize, batchSize)
      val=dataGenerator("validate.wav", chunkSize, batchSize)
      model=makeModel(chunkSize, lstmSize, secondLSTM, secondDNN)
      early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
      history = model.fit(train, validation_data=val, epochs=50, callbacks=[early_stopping], verbose=2)
      valLoss = history.history['val_loss'][-1]
      if valLoss<bestLoss:
       bestLoss=valLoss
       bestParams=[chunkSize,batchSize,lstmSize,secondLSTM,secondDNN]
      print(f"Best loss: {bestLoss}   bestParams: {bestParams}\n\n")
 exit()
 '''


 '''
 chunkSize=64
 batchSize=40
 train=dataGenerator("train.wav", chunkSize, batchSize)
 val=dataGenerator("validate.wav", chunkSize, batchSize)
 test=dataGenerator("test.wav", chunkSize, batchSize)
 model = makeModel(chunkSize, lstmSize=500, learningRate=5e-03)
 model.fit(train, validation_data=val, epochs=4)
 print(model.evaluate(test))
 model.save("completeModel.h5")
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