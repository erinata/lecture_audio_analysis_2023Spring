import librosa
import pandas

import glob
import numpy

import matplotlib.pyplot as pyplot

import os

genres = ["blues","country","hiphop","metal","reggae","classical","disco","jazz","pop","rock"]

if not os.path.exists("chromagram_dataset"):
  os.mkdir("chromagram_dataset")
if not os.path.exists("mfcc_dataset"):
  os.mkdir("mfcc_dataset")
if not os.path.exists("spectrogram_dataset"):
  os.mkdir("spectrogram_dataset")
  
  

for genre in genres:
  print(genre)
  if not os.path.exists("chromagram_dataset/" + genre):
    os.mkdir("chromagram_dataset/" + genre)
  if not os.path.exists("mfcc_dataset/" + genre):
    os.mkdir("mfcc_dataset/" + genre)
  if not os.path.exists("spectrogram_dataset/" + genre):
    os.mkdir("spectrogram_dataset/" + genre)
  for filename in glob.glob("./genres/" + genre + "/*.wav"):
    print(filename, " processing.")
    num = filename.split("/")[-1].split(".")[1]
    x, sr = librosa.load(filename, sr=44100)
    
    stft_data = librosa.stft(y=x)
    stft_data_db = librosa.amplitude_to_db(abs(stft_data))
    spectrogram = pyplot.figure(figsize=(stft_data.shape[1], stft_data.shape[0]), frameon=False, dpi=1)
    librosa.display.specshow(stft_data_db, cmap='gray', sr=sr)
    pyplot.axis('off')
    spectrogram.savefig("spectrogram_dataset/" + genre + "/" + num + ".png")
    pyplot.close()
    
    chroma_stft_data = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=int(sr/2))
    # print(chroma_stft_data.shape)
    chromagram = pyplot.figure(figsize=(chroma_stft_data.shape[1], chroma_stft_data.shape[0]), frameon=False, dpi=1)
    pyplot.axis('off')
    librosa.display.specshow(chroma_stft_data, hop_length=int(sr/2), cmap='gray')
    chromagram.savefig("chromagram_dataset/" + genre + "/" + num + ".png")
    pyplot.close()
    
        
    mfcc_data = librosa.feature.mfcc(y=x, sr=sr, hop_length=int(sr/2))
    # print(mfcc_data.shape)
    mfcc_plot = pyplot.figure(figsize=(mfcc_data.shape[1], mfcc_data.shape[0]), frameon=False, dpi=1)
    pyplot.axis('off')
    librosa.display.specshow(mfcc_data, hop_length=int(sr/2), cmap='gray')
    mfcc_plot.savefig("mfcc_dataset/" + genre + "/" + num + ".png")
    pyplot.close()



            
        