import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


#audio_path = '../T08-violin.wav'
audio_path = 'genres/blues/blues.00000.au'

# the default sampling rate is 22KHz
# can change the sampling rate by calling librosa.load(audio_path, sr=44000)
# or librosa.load(audio_path, sr=None) to avoid re-sampling
x , sr = librosa.load(audio_path)

print(type(x), type(sr))
#<class 'numpy.ndarray'> <class 'int'>

print(x.shape, sr)
#(396688,) 22050


ipd.Audio(audio_path)



# to plot the amplitude envelope of a waveform:
plt.figure(1, figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
#plt.show()

# Zooming in
n0 = 9000
n1 = 9100

plt.figure(2, figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.show()

# by manually inspecting, I saw that there were 16 zero crossings

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))								  # this should be 16



'''
# to plot the spectogram of a wavefile;
# a spectogram is a visual representation of the spectrum of frequencies of a
# signal as it varies with time.

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))

#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')	# non-log scale
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')  # log scale

plt.colorbar()
plt.show()
'''


'''
# to output a numpy array to a .wav file

librosa.output.write_wav('Outputs/example.wav', x, sr)
'''


'''
# to write our own audio file;
# this is an example tone of 220 Hz

sr = 22050 # sample rate
T = 5.0    # seconds
t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 0.5*np.sin(2*np.pi*220*t)# pure sine wave at 220 Hz

# Playing the audio
ipd.Audio(x, rate=sr) # load a NumPy array

# Saving the audio
librosa.output.write_wav('Outputs/tone_220.wav', x, sr)
'''



