from flask import Flask, render_template, request
from flask.json import jsonify
import os
import tempfile
import librosa
import numpy as np
import time
import math
import requests

import demo  # set PYTHONPATH
from demo import DTYPE, DTYPE_CPU, LOW_PASS_FILTER, use_gpu

import torch
from torch.autograd import Variable


app = Flask(__name__)

SAMPLE_RATE = 22050
BATCH_SIZE = 1
STATE_FILE = "data/model_87008c9_99_000000889.state"

dataset = demo.BoatDataset("../data.csv", "../sounds-spectrogram.pkl", test_set=True)
INPUT_SIZE = dataset[0][0].size(1)
print "state file:", STATE_FILE
print "input size:", INPUT_SIZE
print "batch size:", BATCH_SIZE

has_alerted = set()

# make net
net = demo.BoatNet(INPUT_SIZE)
if use_gpu:
    net.cuda()
net.train(False)

# set weights
if use_gpu:
    state_dict = torch.load(STATE_FILE)
else:
    # load but convert storage to CPU
    # see: https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
    state_dict = torch.load(STATE_FILE, map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/save', methods=['GET', 'POST'])
def save():
    t0 = time.time()
    threshold = float(request.args.get("threshold", 0.95))
    location_id = int(request.args.get("id", 0))
    f = request.files['audio-blob']  # get file object
    fd, fn = tempfile.mkstemp(suffix=".wav")
    os.fdopen(fd).close()  # close open file
    f.save(fn)  # save file to temp location

    # read wav file and process
    audio, rate = librosa.load(fn)

    # clean up temp file
    os.unlink(fn)

    # resample to a set rate
    audio = librosa.core.resample(audio, rate, SAMPLE_RATE)

    audio_spec = librosa.feature.melspectrogram(y=audio)
    audio_spec = audio_spec[LOW_PASS_FILTER, :].copy()

    y_pred, y_pred_scores = evaluate_audio_spec(audio_spec)
    t = time.time() - t0

    y_pred = y_pred[0]
    prediction = (y_pred in [1, 2])
    score = math.e ** y_pred_scores[0][y_pred]
    print "Predicted: %s (score=%.3f) in %.3fs" % (y_pred, score, t)

    # submit to poachstopper, if detected
    label = "Nothing special"
    if prediction and score > threshold:
        label = "Midwater Trawler" if y_pred == 1 else "Deepwater Trawler"
        alert_poachstopper(location_id, label)

    return jsonify(
        prediction=prediction,
        label_id=y_pred,
        score=score,
        time=t,
        threshold=threshold,
        label=label
    )


def evaluate_audio_spec(audio_spec):

    # make tensors
    x = audio_spec.transpose(1, 0)
    x = torch.Tensor(x).type(DTYPE_CPU)
    x = x.unsqueeze(0)  # add batch dimension
    if use_gpu:
        x = x.cuda()

    # get predictions
    hidden = net.init_hidden(batch=BATCH_SIZE)  # reset hidden layer!
    hidden.volatile=True  # set to volatile since don't care about gradient
    y_pred_scores = net(Variable(x, volatile=True), hidden).data
    y_pred = y_pred_scores.cpu().numpy().argmax(axis=1).tolist()

    return y_pred, y_pred_scores


def alert_poachstopper(location_id=0, label="Midwater Trawler"):
    if location_id in has_alerted:
        print "Skipping alerting a second time for location_id", location_id
        return

    # The Georges bank locations are:
    if location_id == 1:
        lat, lon = 41.25, -67.5
    else:
        lat, lon = 40.9, -68.0

    payload = {
        "location[title]": "Sensor %d" % location_id,
        "location[latitude]": lat,
        "location[longitude]": lon,
        "location[description]": label,
        "location[status]": "Violation!",
        "commit": "Add Location"
    }

    response = requests.post("http://www.poachstopper.net/locations", payload)
    has_alerted.add(location_id)
    print "Posted to poachstopper [%d]" % response.status_code  # should be 200
