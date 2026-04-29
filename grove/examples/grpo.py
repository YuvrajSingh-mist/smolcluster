"""GRPO fine-tuning with verifier rewards.

Each step: generate K completions, score with a reward function,
compute advantages, train with policy gradient. CPU-parallel
verification leverages Apple Silicon's unified memory — zero-copy
between GPU (generation) and CPU (verification).

    grove run examples/grpo.py                # single device
    grove start examples/grpo.py -n 2         # distributed
"""

import grove
import random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from concurrent.futures import ThreadPoolExecutor


K = 4
STEPS = 100


class TinyLM(nn.Module):
    def __init__(self, vocab=256, dim=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.head = nn.Linear(dim, vocab)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = nn.gelu(layer(h))
        return self.head(h)

    def generate(self, prompt_tokens, max_tokens=16):
        tokens = list(prompt_tokens)
        for _ in range(max_tokens):
            logits = self(mx.array([tokens]))[:, -1, :]
            probs = mx.softmax(logits / 0.8, axis=-1)
            next_tok = mx.random.categorical(probs).item()
            tokens.append(next_tok)
        return tokens


def reward_fn(completion):
    """Reward: higher diversity of tokens = better. CPU-bound scoring."""
    unique = len(set(completion))
    return unique / len(completion)


def main():
    world = grove.init()
    mx.random.seed(42 + world.rank())
    random.seed(world.rank())

    model = TinyLM()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=1e-4)
    diloco = grove.sparseloco(model, H=10, outer_lr=0.5, topk=32, chunk=64)

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    print(f"rank {world.rank()}: {n_params:,} params")

    prompts = [[i % 256, (i * 7) % 256, (i * 13) % 256] for i in range(200)]
    my_prompts = prompts[world.rank()::world.size()]

    for step in range(STEPS):
        prompt = my_prompts[step % len(my_prompts)]

        completions = [model.generate(prompt) for _ in range(K)]
        mx.eval(model.parameters())

        with ThreadPoolExecutor(max_workers=K) as pool:
            rewards = list(pool.map(reward_fn, completions))

        mean_r = np.mean(rewards)
        std_r = max(np.std(rewards), 1e-6)
        advantages = [(r - mean_r) / std_r for r in rewards]

        def grpo_loss(model):
            total = mx.array(0.0)
            for comp, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                logits = model(mx.array([comp[:-1]]))
                log_p = -nn.losses.cross_entropy(logits, mx.array([comp[1:]]), reduction="mean")
                total = total + log_p * adv
            return -total / K

        loss, grads = nn.value_and_grad(model, grpo_loss)(model)
        mx.eval(loss, grads)
        optimizer.update(model, grads)
        mx.eval(model.state, optimizer.state)

        synced = diloco.step(model)
        grove.report(loss.item())

        if world.rank() == 0 and (step % 10 == 0 or synced):
            tag = " [sync]" if synced else ""
            print(f"  step {step}: loss={loss.item():.4f} reward={mean_r:.3f}{tag}")

    if world.rank() == 0:
        print(f"  done ({diloco.outer_step} syncs)")


if __name__ == "__main__":
    main()
