import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import sklearn
import csv



genres = ['blues', 'classical', 'country', 'disco', 'hiphop', \
				  'jazz', 'metal', 'pop', 'reggae', 'rock']


GeneratePlots = False
c = csv.writer(open("AllDataFeatures_trimmed.csv", "w"))


# Normalizing the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def GenerateCSVs(audio_path):
	x, sr = librosa.load(audio_path)

	# get the genre name out of the audio path
	a,genre_name,d = audio_path.split('/')
	genre_index = int(genres.index(genre_name))


	# 1. Calculate the number of zero crossings
	zero_crossings = librosa.zero_crossings(x, pad=False)
	sumZeroCrossings = sum(zero_crossings)

	# 2. Calculate the spectral centroid
	spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

	# 3. Calculate spectral roll-off
	spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

	# 4. Calculate Mel-Frequency Cepstral Coefficients
	mfccs = librosa.feature.mfcc(x)
	mfccs = sklearn.preprocessing.scale(mfccs, axis=1)

	# 5. Calculate Chroma Frequencies
	hop_length = 512
	chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

	# trim all of the matrices to be a smaller length (to compensate for minor
	# size differences)
	small_len = 1285

	spectral_centroids = spectral_centroids[:small_len]
	spectral_rolloff   = spectral_rolloff[:small_len]
	mfccs              = mfccs[:, :small_len]
	chromagram 				 = chromagram[:, :small_len]


	'''
	# debugging aid:  print out shapes of each of the features
	print("   spectral_centroids len:", spectral_centroids.shape)
	print("   spectral_rolloff len:", spectral_rolloff.shape)
	print("   mfccs len:", mfccs.shape)
	print("   chromagram len:", chromagram.shape)
	print()
	'''

	# write the data
	nextRow = np.hstack([genre_index, sumZeroCrossings, spectral_centroids,
		                 np.hstack(spectral_rolloff), np.hstack(mfccs), np.hstack(chromagram)])
	c.writerow(nextRow)


	# if we want to generate plots, then do so
	if GeneratePlots == True:

		# plot the amplitude envelope of a waveform:
		plt.figure(1, figsize=(14, 5))
		librosa.display.waveplot(x, sr=sr)
		title = audio_path + " Waveform"
		plt.title(title)

		# to plot the spectogram of a wavefile
		X = librosa.stft(x)
		Xdb = librosa.amplitude_to_db(abs(X))
		plt.figure(2, figsize=(14, 5))
		title = audio_path + " Spectogram Visualization"
		plt.title(title)
		#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')	# non-log scale
		librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')  # log scale
		plt.colorbar()

		# 2. spectral centroid
		frames = range(len(spectral_centroids))
		t = librosa.frames_to_time(frames)
		plt.figure(3, figsize=(14, 5))
		librosa.display.waveplot(x, sr=sr, alpha=0.4)
		plt.plot(t, normalize(spectral_centroids), color='r')
		title = audio_path + " Spectral Centroids"
		plt.title(title)

		# 3. spectral roll-off
		plt.figure(4)
		librosa.display.waveplot(x, sr=sr, alpha=0.4)
		plt.plot(t, normalize(spectral_rolloff), color='r')
		title = audio_path + " Spectral Roll-off"
		plt.title(title)

		# 4. MFCC
		plt.figure(5)
		librosa.display.specshow(mfccs, sr=sr, x_axis='time')
		title = audio_path + " MFCC"
		plt.title(title)


		# 5. chroma frequencies
		plt.figure(6, figsize=(15, 5))
		librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
		title = audio_path + " Chroma Frequencies"
		plt.title(title)

		plt.show()



AudioPaths = []

Genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
Numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for genre in Genres:
	for firstDigit in Numbers:
		for secondDigit in Numbers:
			pathName = 'genres/' + genre + '/' + genre + '.000' + firstDigit + secondDigit + '.au'
			AudioPaths.append(pathName)





for file in AudioPaths:
	print('Analyzing', file, '...')
	GenerateCSVs(file)








