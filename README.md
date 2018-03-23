# Fishackathon demo server

## Run demo

Install requirements:

```
torch
numpy
flask
librosa
matplotlib
requests
```

For example, to run demo:

```
cd server
PYTHONPATH="$PYTHONPATH:$HOME/fishackathonr-refactor" \
  FLASK_DEBUG=0 \
  FLASK_APP=app.py \
  flask run --port 9082
```

## Training

Download sounds training/testing data

```
./download-data.sh
```

Edit `demo.py` as required, then run:

```
./demo.py
```

## Evaluating

See `Eval.py`
