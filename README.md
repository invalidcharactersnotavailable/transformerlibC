# transformerlibC

## usage

```
make
./transformer_test [--data <file>] [--save <ckpt>] [--load <ckpt>] [--test]
```

- `--data <file>`: path to text file, one line per sample
- `--save <ckpt>`: save model checkpoint after training
- `--load <ckpt>`: load model checkpoint before training
- `--test`: run a forward+backward test/benchmark and exit

## configuration

- set hyperparameters in `main.c` (layers, embed_dim, etc)
- for 100m+ param models, ensure enough RAM is available

## scaling

- no hardcoded parameter limits
- supports float32 and float16 (mixed precision)
- optimizer: sgd and adam
- autodiff: efficient, iterative, dynamic

## features

- real optimizer logic (sgd/adam)
- mixed precision (float16/float32)
- efficient autodiff
- error handling, logging, input validation
- checkpointing (save/load)
- real dataset batching, shuffling
- test/benchmark mode

## requirements

- gcc (with -O3, -fopenmp)
- cblas (optional, for BLAS matmul)
- enough RAM for your model size

## notes

- this is a minimal educational framework, not a full production system
- for real use, extend with distributed training, advanced data loading, and more tests
