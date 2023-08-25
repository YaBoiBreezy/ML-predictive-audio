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
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

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


if __name__ == "__main__":
 '''
 chunkSize=64
 model = tf.keras.models.load_model('completeModel.h5')
 #22050 hz, 43.3 spectrogram frames per second
 test = readWavFile("test.wav") [:22050*5] #<<======================================================
 melData = librosa.feature.melspectrogram(y=test, sr=fs, hop_length=512) #create mel spectrogram
 maxer = np.max(melData)
 logMelData = librosa.power_to_db(melData, ref=maxer)
 dbMin=np.min(logMelData)
 normalizedData = logMelData / dbMin
 data = np.transpose(normalizedData)
 '''



 train = readWavFile("train.wav") [:22050*5]
 train_size = int(0.8 * len(train))
 train, test = train[:train_size], train[train_size:]

 # Convert numpy arrays to pandas DataFrame with appropriate time index
 train_index = pd.date_range(start='2023-01-01', periods=len(train), freq='T')  # Adjust as needed
 test_index = pd.date_range(start='2023-01-01', periods=len(test), freq='T')    # Adjust as needed

 train_df = pd.DataFrame({'target': train}, index=train_index)
 test_df = pd.DataFrame({'target': test}, index=test_index)

 # Create the Exponential Smoothing model
 model = ExponentialSmoothing(train_df['target'], seasonal='add', seasonal_periods=12)

 # Fit the model to the training data
 model_fit = model.fit()

 # Define the forecast horizon
 forecast_steps = len(test_df)

 # Make predictions
 forecast = model_fit.predict(start=len(train_df), end=len(train_df) + forecast_steps - 1)

 # Plot the actual and predicted values
 plt.plot(train_df.index, train_df['target'], label='Train')
 plt.plot(test_df.index, test_df['target'], label='Test', color='blue')
 plt.plot(forecast.index, forecast, label='Forecast', color='red')
 plt.legend()
 plt.show()
 exit()














 data1=data
 for i in range(64, len(data1)-2, 2):
  data1[i+1] = data1[i]
 data1 = data1 * dbMin
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data1), ref=maxer))
 playWav(rawData)

 data2=data
 for i in range(64, len(data2)-2, 2):
  predict=model.predict(np.array([data2[i-64:i]]), verbose=0)
  data2[i+1]=predict
 data2 = data2 * dbMin
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data2), ref=maxer))
 playWav(rawData)
 exit()
 '''
 for _ in range(43*2):
  predict=model.predict(np.array([data[-64:]]), verbose=0)
  l2Norm = np.linalg.norm(predict)
  print(data[-3])
  data=np.concatenate((data,predict/l2Norm), axis=0)
 '''
 data = data * dbMin
 print(data)
 rawData=librosa.feature.inverse.mel_to_audio(librosa.db_to_power(np.transpose(data), ref=maxer))
 print(rawData)
 print(data.shape)
 print(rawData.shape)
 playWav(rawData)
 display(data)
