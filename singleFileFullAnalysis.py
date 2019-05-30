import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import sklearn


audio_path = 'genres/classical/classical.00000.au'

# the default sampling rate is 22KHz
# can change the sampling rate by calling librosa.load(audio_path, sr=44000)
# or librosa.load(audio_path, sr=None) to avoid re-sampling
x , sr = librosa.load(audio_path)

print(type(x), type(sr))
#<class 'numpy.ndarray'> <class 'int'>

print(x.shape, sr)
#(396688,) 22050


#ipd.Audio(audio_path)



# to plot the amplitude envelope of a waveform:
plt.figure(1, figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
title = audio_path + " Waveform"
plt.title(title)


# to plot the spectogram of a wavefile;
# a spectogram is a visual representation of the spectrum of frequencies of a
# signal as it varies with time.

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(2, figsize=(14, 5))
title = audio_path + " Spectogram Visualization"
plt.title(title)

#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')	# non-log scale
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')  # log scale
plt.colorbar()
#plt.show()



# 1. Calculate the number of zero crossings
zero_crossings = librosa.zero_crossings(x, pad=False)
print("zero_crossings shape:"),
print(zero_crossings.shape)

sumZeroCrossings = sum(zero_crossings)
print(sumZeroCrossings)


# 2. Calculate the spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
print("spectral_centroids shape:"),
print(spectral_centroids.shape)

# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalizing the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Plotting the Spectral Centroid along the waveform
plt.figure(3, figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
title = audio_path + " Spectral Centroids"
plt.title(title)


# 3. Calculate spectral roll-off
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
print("spectral_rolloff shape:"),
print(spectral_rolloff.shape)
plt.figure(4)
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
title = audio_path + " Spectral Roll-off"
plt.title(title)


# 4. Calculate Mel-Frequency Cepstral Coefficients
mfccs = librosa.feature.mfcc(x)
print("mfccs shape:"),
print(mfccs.shape)
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
#print(mfccs.mean(axis=1))
#print(mfccs.var(axis=1))
plt.figure(5)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
title = audio_path + " MFCC"
plt.title(title)


# 5. Calculate chroma frequencies
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
print("chromagram shape:"),
print(chromagram.shape)
plt.figure(6, figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
title = audio_path + " Chroma Frequencies"
plt.title(title)



plt.show()



