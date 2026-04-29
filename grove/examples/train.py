"""Distributed LoRA fine-tuning.

    grove start examples/train.py -n 4
"""

import grove
import json
import random
import sys
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers


def main():
    world = grove.init()

    model, tokenizer = load("mlx-community/SmolLM2-135M-Instruct")
    linear_to_lora_layers(
        model, num_layers=4,
        config={"rank": 8, "scale": 1.0, "dropout": 0.0},
    )
    model.freeze()
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_a"):
            mod.unfreeze()
    mx.eval(model.parameters())

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"rank {world.rank()}: {n_params:,} trainable params")

    data = []
    for a in range(1, 100):
        for b in range(1, 100):
            data.append(
                f"<|im_start|>user\nWhat is {a} + {b}?<|im_end|>\n"
                f"<|im_start|>assistant\n{a + b}<|im_end|>"
            )
    random.seed(world.rank() + 42)
    random.shuffle(data)
    my_data = data[world.rank()::world.size()]

    optimizer = optim.Adam(learning_rate=2e-4)
    diloco = grove.diloco(model, H=100, outer_lr=0.3)
    loss_and_grad = nn.value_and_grad(
        model,
        lambda m, x: nn.losses.cross_entropy(m(x[:, :-1]), x[:, 1:], reduction="mean"),
    )

    grove.barrier()

    running_loss = 0.0
    for step in range(300):
        batch = my_data[step * 4:(step + 1) * 4]
        if len(batch) < 4:
            batch = my_data[:4]
        tokens_list = [tokenizer.encode(t)[:128] for t in batch]
        max_len = max(len(t) for t in tokens_list)
        pad_id = tokenizer.eos_token_id or 0
        padded = [t + [pad_id] * (max_len - len(t)) for t in tokens_list]
        tokens = mx.array(padded)

        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)

        synced = diloco.step(model)
        l = loss.item()
        grove.report(l)
        running_loss = 0.9 * running_loss + 0.1 * l if running_loss > 0 else l

        if step % 5 == 0:
            print(json.dumps({
                "type": "stats",
                "rank": world.rank(),
                "step": step,
                "loss": round(running_loss, 4),
                "synced": synced,
            }))
            sys.stdout.flush()

        if world.rank() == 0 and step % 20 == 0:
            tag = " [sync]" if synced else ""
            print(f"  step {step}: loss={running_loss:.4f}{tag}")

    print(f"rank {world.rank()}: done")
