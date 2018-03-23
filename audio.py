# preprocessing of .wav and .ogg sound files to a common format
# then return different spectral and audio representations
import os
import glob
import cPickle as pickle

import librosa
import numpy as np


SOUNDS_DIRS = [
    "../projects/Fishackathon/sounds/*",
    "../data/longfish/sounds/*.wav"
]


def audio_spectrogram(audio_file, sample_rate=22050, log_scale=False):
    audio, rate = librosa.load(audio_file)
    
    # resample to a set rate
    audio = librosa.core.resample(audio, rate, sample_rate)

    audio_spec = librosa.feature.melspectrogram(y=audio)
    
    if log_scale:
        # Convert to log scale (dB). We'll use the peak power (max) as reference.
        audio_spec = librosa.power_to_db(audio_spec, ref=np.max)

    # return audio spectrogam that as a matrix
    return audio_spec


if __name__ == '__main__':

    # process and cache sounds
    audio_files = []
    for path in SOUNDS_DIRS:
        audio_files += glob.glob(path)

    n = len(audio_files)
    cache = {}
    for i, fn in enumerate(audio_files):
        k = os.path.basename(fn)
        print "processing", i + 1, 'of', n, k,
        s = audio_spectrogram(fn, log_scale=True)
        print s.shape
        cache[k] = s

    with open("sounds-spectrogram-log.pkl", "wb") as f:
        pickle.dump(cache, f, -1)

    print n, len(cache)
