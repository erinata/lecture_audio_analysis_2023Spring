import librosa
import numpy

import matplotlib.pyplot as pyplot


genre = "jazz"
num = "00008"


audio_file = "genres/" + genre + "/" + genre + "." + num + ".wav"
x, sr = librosa.load(audio_file, sr=44100)

wave_plot=pyplot.figure(figsize=(13,5))
librosa.display.waveshow(y=x, sr=sr)
wave_plot.savefig("temp_plots/waveplot" + genre + num + ".png")
pyplot.close()

spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
print(spectral_centroids)
spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sr)[0]
print(spectral_rolloff)
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
print(spectral_bandwidth)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x)
print(zero_crossing_rate)
mfcc = librosa.feature.mfcc(y=x, sr=sr)
print(mfcc)


stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram =pyplot.figure(figsize=(13,5))
librosa.display.specshow(stft_data_db, sr=sr, x_axis="time", y_axis="hz")
pyplot.colorbar()
spectrogram.savefig("temp_plots/spectrogram_" + genre + num +".png")
pyplot.close()


stft_data = librosa.stft(x)
stft_data_db = librosa.amplitude_to_db(abs(stft_data))
spectrogram = pyplot.figure(figsize=(stft_data.shape[1], stft_data.shape[0]), frameon=False, dpi=1)
librosa.display.specshow(stft_data_db, sr=sr, cmap="gray")
pyplot.axis('off')
spectrogram.savefig("temp_plots/gray_spectrogram_" + genre + num +".png")
pyplot.close()




chroma_stft_data = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=sr)
chromagram = pyplot.figure(figsize=(13,5))
librosa.display.specshow(chroma_stft_data, x_axis="time", y_axis="chroma", hop_length=sr, cmap='coolwarm')
chromagram.savefig("temp_plots/chromagram_" + genre + num + ".png")
pyplot.close()

chroma_stft_data = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=int(sr/2))
print(chroma_stft_data.shape)
chromagram = pyplot.figure(figsize=(chroma_stft_data.shape[1], chroma_stft_data.shape[0]), frameon=False, dpi=1)
pyplot.axis('off')
librosa.display.specshow(chroma_stft_data, hop_length=int(sr/2), cmap='gray')
chromagram.savefig("temp_plots/gray_chromagram_" + genre + num + ".png")
pyplot.close()





mfcc_data = librosa.feature.mfcc(y=x, sr=sr)
mfcc_plot = pyplot.figure(figsize=(13,5))
librosa.display.specshow(mfcc_data, x_axis="time")
mfcc_plot.savefig("temp_plots/mfcc_" + genre + num + ".png")
pyplot.close()


mfcc_data = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(sr/2))
print(mfcc_data.shape)
mfcc_plot = pyplot.figure(figsize=(mfcc_data.shape[1], mfcc_data.shape[0]), frameon=False, dpi=1)
pyplot.axis('off')
librosa.display.specshow(mfcc_data, hop_length=int(sr/2), cmap='gray')
mfcc_plot.savefig("temp_plots/gray_mfcc_" + genre + num + ".png")
pyplot.close()













