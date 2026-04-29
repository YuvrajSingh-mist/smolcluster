# grove

Distributed ML training across MacBooks. Zero config.

```bash
pip install grove-ml
```

**Mac A:**
```bash
grove start train.py -n 2
```

**Mac B:**
```bash
grove join
```

Both machines discover each other automatically, sync gradients, and train together. No SSH, no IP addresses, no configuration files.

Grove discovers peers over AWDL (the protocol behind AirDrop), then upgrades to direct WiFi when both devices share a network. If WiFi isn't available (e.g. eduroam, or no network at all), everything stays on AWDL.

## Quick start

Write a training script with a `main()` function:

```python
# train.py
import grove
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def main():
    world = grove.init()

    model = nn.Linear(64, 64)
    optimizer = optim.SGD(learning_rate=0.01)

    for step in range(100):
        x = mx.random.normal((8, 64))
        y = mx.random.normal((8, 64))

        loss, grads = nn.value_and_grad(model, lambda m, x, y: mx.mean((m(x) - y) ** 2))(model, x, y)
        grads = grove.average_gradients(grads)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)
```

Single device:
```bash
grove run train.py
```

Multiple devices:
```bash
grove start train.py -n 2    # coordinator
grove join                    # worker (shows interactive picker)
```

Workers receive the training script from the coordinator automatically.

## Algorithms

### DiLoCo

Each device trains independently for H steps, then syncs pseudo-gradients with Nesterov momentum. Good default for most setups.

```python
diloco = grove.diloco(model, H=500, outer_lr=0.7)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.state, optimizer.state)
    diloco.step(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H` | 500 | Inner steps between syncs |
| `outer_lr` | 0.7 | Outer optimizer learning rate |
| `outer_momentum` | 0.9 | Nesterov momentum |
| `overlap` | False | Async overlap (sync in background) |
| `quantize` | False | E3M0 4-bit pseudo-gradients |

### SparseLoCo

DiLoCo with top-k compression and error feedback. Sends only the largest 1-3% of values each round, with unsent values carrying forward. ~32x less communication than dense DiLoCo.

```python
sloco = grove.sparseloco(model, H=500, topk=64, chunk=4096)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.state, optimizer.state)
    sloco.step(model)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `H` | 30 | Inner steps between syncs |
| `outer_lr` | 1.0 | Outer optimizer learning rate |
| `topk` | 64 | Values kept per chunk |
| `chunk` | 4096 | Chunk size for top-k selection |
| `error_decay` | 0.95 | Decay on error buffer |
| `overlap` | True | Async overlap (on by default) |

### DeMo

DCT-compressed per-step sync. Transforms gradients to frequency space and sends the most significant components. Syncs every step rather than every H steps. Better suited for fast local networks.

```python
demo = grove.demo(model, lr=1e-3, topk=32)

for step in range(total_steps):
    loss, grads = loss_and_grad(model, batch)
    demo.step(model, grads)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `decay` | 0.999 | EMA decay |
| `topk` | 32 | DCT components kept per chunk |
| `chunk` | 64 | Chunk size |

## API

### Initialization

```python
world = grove.init()
world.rank()   # this device's rank (0 = coordinator)
world.size()   # total number of devices
```

### Collective operations

```python
grove.average_gradients(grads)  # all-reduce + average
grove.all_sum(x)                # sum an MLX array across devices
grove.all_gather(x)             # gather an MLX array from all devices
grove.send(x, dst)              # send to a specific rank
grove.recv(shape, dtype, src)   # receive from a specific rank
grove.barrier()                 # wait for all devices
grove.report(loss)              # report loss to dashboard
```

### Status

```python
grove.rank          # int
grove.world_size    # int
grove.is_available() # True if world_size > 1
```

## CLI

```
grove run <script>              Run on a single device
grove start <script> -n N       Start a cluster with N nodes
grove start <script> --name X   Start with a specific cluster name
grove join [name]               Join a cluster (interactive picker if no name)
grove status                    System info and nearby clusters
```

Add `--logs` to any command to see raw log output instead of the dashboard.

## Environment variables

| Variable | Effect |
|----------|--------|
| `GROVE_NO_WIFI` | Skip WiFi upgrade probe, use AWDL only |

## Requirements

- macOS with Apple Silicon (M1+)
- Python 3.10+
- [MLX](https://github.com/ml-explore/mlx)
- Xcode command-line tools (for compiling the Swift helper on first run)

## License

MIT
