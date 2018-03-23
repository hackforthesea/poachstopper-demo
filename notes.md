 - #1 train network
     - [x] use sounds.pkl
     - [x] implement GRU network
     - [x] train and visualize
     - [x] evaluate
     - [x] confusion matrix

 - #2 resample data
     - [x] use spectogram directly
     - [x] create training / test subsets
     - [x] low-pass filter slice(0:63) of 128
     - [x] try labels for each individual class at training (not significantly different in accuracy)

 - #3 mix data for augmentation
     - [x] train with GPU (p2) <<<
         - batch utilization ~%50; memory ~200Mb - 500mb (of 12); ~2GB of host mem of 60GB
         - 36s --> 7s per batch size 64 (5.1x)
         - 58s --> 8.7s  "   128 (6.6x)
         - 106 --> 13   "    256 (8.2x)
         - 209 --> 22    "   512 (9.5x)  <--
         - 267 -->  32    " 1024 (8.3x)
     - [x] how much time spent loading data?
           (significant amount; loading data using cpu workers dramatically increases speed)
     - [x] pinned memory (only applies to CPU memory)
         - 22s --> 14s  by changing dataloader to output pinned-memory CPU tensors (512 batch),
           and converting before use; 50% --> %60 GPU utilization
         - --> 9.8s by increasing dataloader `num_workers` from 0 to 4 (or 8) (75% utilization)
           (which is 209./9.8:   21.3x !)
         - increasing batch size again from 512 --> 1024: 12s (vs 32) or 22.3x speedup; 82% util.
         - 2048 batch size --> 88% util
         - 4096  --> 93% util with 16 dataloader workers
           (still doesn't get close to full memory util on p2)
         - NOTE: loss does not go down as fast with the larger batch (because optimizer step is
           counted in terms of number of backward steps; increase learning rate)
         - 8000  --> 98% utilization (1/3 of GPU mem used) 11 seconds for full epoch
         - final experiment: 307s --> 11s per epoch **(36.1x)** !!
     - [x] monitor / optimize GPU utilization (get to 100%); best practices?
     - [x] train with nn.DataParallel on CPU (instead of multi-processing) (didn't see difference)
     - [ ] train with nn.DataParallel on GPU <<<
     - [ ] train with multi GPU <<<
     - [ ] train with torch.multiprocessing (e.g., Hogwild) <<<

 - #4 experiment with architecture / training
     - [ ] blend background samples (and ambient noise from cafes etc)
     - [ ] experiment with architecture
     - [ ] experiment with regularization
     - [ ] experiment with hyper-parameters
     - [ ] best practices for logging accuracy and metrics
     - [ ] 2D convolutional layer over fixed time window
     - [ ] vector embedding with triplet loss

 - #5 write demo server
     - [x] train / pick model
     - [x] write javascript to listen to mic and post to server
     - [x] write flask app to get mic sounds, process, evaluate and return to client
     - [x] if works, hook up with demo front end

 - other
     - [ ] try Git LFS (with S3 backend?)
     - [ ] best practices for monitoring / optimizing GPU performance?

---

issues:
 - DataLoader multiprocessing (Dataset pickling)
 - Process of synchronizing model state with code that generated it
     - code
     - dataset (labels, bytes, clean-up errors, multi-class, augmentation)
     - model
     - evaluation code, graphs, visualizations
     - temporary glue code / data cleanup code
     - documentation
 - "Cannot re-initialize CUDA in forked subprocess."  (dataloader? module-level statements?)

IDEA: "timeline view" of a project:
 - sync documentation, source code, local working directory, temporary scripts
   to make a process reproducible.
 - "demo.py:53: UserWarning: RNN module weights are not part of single contiguous chunk of memory.
   This means they need to be compacted at every call,
   possibly greately increasing memory usage. To compact weights again call `flatten_parameters()`."
 - make sure Module weights are moved to gpu by using _Module_`.cuda()`

---

interesting:
https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/

http://pytorch.org/docs/master/multiprocessing.html#module-torch.multiprocessing

visualization
 - pytorch TNT (aka torchnet)!   `pip install --upgrade git+https://github.com/pytorch/tnt.git@master`
 - ignite (pytorch project to provide flexible training engine)
 - tensorboard (google)
 - visdom (facebook)
 - ipython notebook

```
pip install --upgrade git+https://github.com/pytorch/tnt.git@master
python -m visdom.server -port 9080 # start server
```

model_00_000002999.state   (fixed hidden layer bug)

    got 948 of 1000: %94.80
    saw Counter({2: 537, 1: 334, 0: 129})

models-03-02-18/model_00_000001599.state

    got 927 of 1000: %92.70
    saw Counter({2: 537, 1: 334, 0: 129})
    demo.py
