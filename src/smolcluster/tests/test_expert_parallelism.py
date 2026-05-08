"""
Pytest test suite for Expert Parallelism (EP) components.

Tests cover:
- get_expert_per_node: expert shard assignment per rank
- get_ep_components_per_node: rank-based model component initialization
- route_tokens: routing activations to per-expert buckets
- ExpertBlock: forward shape and gradient flow with sharded dims
- Router: probability and index output shapes
- End-to-end EP pipeline: TextEmbeddings → Router → ExpertBlocks → Mixtral
"""

import pytest
import torch
import torch.nn as nn

from smolcluster.algorithms.ExpertParallelism.worker import compute_expert_contributions, route_tokens
from smolcluster.models.moe import ExpertBlock, Mixtral, Router, TextEmbeddings
from smolcluster.utils import get_expert_per_node, get_model_per_node


# ---------------------------------------------------------------------------
# get_expert_per_node
# ---------------------------------------------------------------------------

class TestGetExpertPerNode:

    def test_experts_cover_all_indices(self, ep_num_nodes, ep_num_experts):
        """Every expert index appears in exactly one rank's shard."""
        all_assigned = []
        for rank in range(ep_num_nodes):
            all_assigned.extend(get_expert_per_node(rank, ep_num_nodes, ep_num_experts))
        assert sorted(all_assigned) == list(range(ep_num_experts))

    def test_shard_sizes_are_equal(self, ep_num_nodes, ep_num_experts):
        """Each rank gets the same number of experts."""
        shard_sizes = [
            len(get_expert_per_node(rank, ep_num_nodes, ep_num_experts))
            for rank in range(ep_num_nodes)
        ]
        assert len(set(shard_sizes)) == 1

    def test_no_overlap_between_ranks(self, ep_num_nodes, ep_num_experts):
        """No expert is assigned to more than one rank."""
        shards = [
            set(get_expert_per_node(rank, ep_num_nodes, ep_num_experts))
            for rank in range(ep_num_nodes)
        ]
        for i in range(ep_num_nodes):
            for j in range(i + 1, ep_num_nodes):
                assert shards[i].isdisjoint(shards[j])

    def test_returns_list_of_ints(self, ep_num_nodes, ep_num_experts):
        shard = get_expert_per_node(0, ep_num_nodes, ep_num_experts)
        assert isinstance(shard, list)
        assert all(isinstance(idx, int) for idx in shard)

    def test_invalid_rank_raises(self, ep_num_nodes, ep_num_experts):
        with pytest.raises(AssertionError):
            get_expert_per_node(ep_num_nodes, ep_num_nodes, ep_num_experts)

    def test_indivisible_experts_raises(self):
        """num_experts not divisible by num_nodes must raise."""
        with pytest.raises(AssertionError):
            get_expert_per_node(0, num_nodes=4, num_experts=6)


# ---------------------------------------------------------------------------
# get_ep_components_per_node
# ---------------------------------------------------------------------------

class TestGetModelPerNodeEP:
    """Tests for get_model_per_node with model_type='causal_mixtral' (EP mode)."""

    def _make_mixtral(self, vocab_size, model_dim, num_heads, num_layers, device):
        return Mixtral(
            vocab_size=vocab_size, embeddings_dims=model_dim, no_of_heads=num_heads,
            no_of_decoder_layers=num_layers, device=device,
        )

    def _cfg(self, ep_top_k, device, noisy_topk=False):
        return dict(top_k=ep_top_k, device=device, noisy_topk=noisy_topk)

    def _call(self, rank, num_nodes, num_experts, num_layers, cfg, mixtral):
        return get_model_per_node(
            model=mixtral, num_nodes=num_nodes, local_rank=rank,
            total_layers=num_layers, model_type="causal_mixtral",
            num_experts=num_experts, model_config=cfg,
        )

    def test_rank0_has_embeddings_and_router(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
        cfg = self._cfg(ep_top_k, device)
        _, layers = self._call(0, ep_num_nodes, ep_num_experts, num_layers, cfg, mixtral)
        assert isinstance(layers["model.text_embeddings"], TextEmbeddings)
        assert isinstance(layers["model.router"], Router)

    def test_middle_rank_has_no_embeddings_or_mixtral(self, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
        cfg = self._cfg(ep_top_k, device)
        _, layers = self._call(rank=1, num_nodes=3, num_experts=6, num_layers=num_layers, cfg=cfg, mixtral=mixtral)
        assert "model.text_embeddings" not in layers
        assert "model.router" not in layers
        assert "model.mixtral" not in layers

    def test_last_rank_has_mixtral(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
        cfg = self._cfg(ep_top_k, device)
        _, layers = self._call(ep_num_nodes - 1, ep_num_nodes, ep_num_experts, num_layers, cfg, mixtral)
        assert isinstance(layers["model.mixtral"], Mixtral)

    def test_all_ranks_have_experts(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        cfg = self._cfg(ep_top_k, device)
        experts_per_rank = ep_num_experts // ep_num_nodes
        for rank in range(ep_num_nodes):
            mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
            _, layers = self._call(rank, ep_num_nodes, ep_num_experts, num_layers, cfg, mixtral)
            expert_keys = [k for k in layers if k.startswith("model.expert_")]
            assert len(expert_keys) == experts_per_rank

    def test_single_node_has_all_components(self, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
        cfg = self._cfg(ep_top_k, device)
        _, layers = self._call(0, 1, ep_num_experts, num_layers, cfg, mixtral)
        assert "model.text_embeddings" in layers
        assert "model.router" in layers
        assert "model.mixtral" in layers
        expert_keys = [k for k in layers if k.startswith("model.expert_")]
        assert len(expert_keys) == ep_num_experts

    def test_modulelist_contains_all_params(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k, small_vocab_size, num_layers, device):
        mixtral = self._make_mixtral(small_vocab_size, ep_model_dim, ep_num_heads, num_layers, device)
        cfg = self._cfg(ep_top_k, device)
        module_list, layers = self._call(0, ep_num_nodes, ep_num_experts, num_layers, cfg, mixtral)
        assert isinstance(module_list, nn.ModuleList)
        assert len(module_list) == len(layers)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestRouter:

    def test_output_shapes(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        x = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, indices = router(x)
        assert probs.shape == (batch_size, small_seq_len, ep_top_k)
        assert indices.shape == (batch_size, small_seq_len, ep_top_k)

    def test_indices_in_range(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        x = torch.randn(batch_size, small_seq_len, ep_model_dim)
        _, indices = router(x)
        assert indices.min() >= 0
        assert indices.max() < ep_num_experts

    def test_probs_sum_to_one(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        x = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, _ = router(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# ExpertBlock
# ---------------------------------------------------------------------------

class TestExpertBlock:

    def test_forward_shape(self, ep_model_dim, ep_num_experts, device, batch_size):
        expert_dim = ep_model_dim // ep_num_experts
        expert = ExpertBlock(expert_dim, device)
        x = torch.randn(batch_size, expert_dim)
        out = expert(x)
        assert out.shape == (batch_size, expert_dim)

    def test_gradient_flow(self, ep_model_dim, ep_num_experts, device, batch_size):
        expert_dim = ep_model_dim // ep_num_experts
        expert = ExpertBlock(expert_dim, device)
        x = torch.randn(batch_size, expert_dim, requires_grad=True)
        out = expert(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# route_tokens
# ---------------------------------------------------------------------------

class TestRouteTokens:

    def test_routed_keys_are_expert_indices(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        activations = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, indices = router(activations)
        all_experts = list(range(ep_num_experts))
        routed, _ = route_tokens(probs, indices, ep_num_experts, all_experts, activations)
        assert all(k in all_experts for k in routed)

    def test_routed_tensor_has_correct_embedding_dim(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        activations = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, indices = router(activations)
        all_experts = list(range(ep_num_experts))
        routed, _ = route_tokens(probs, indices, ep_num_experts, all_experts, activations)
        for expert_idx, tokens in routed.items():
            assert tokens.shape[-1] == ep_model_dim, (
                f"Expert {expert_idx} got tokens with dim {tokens.shape[-1]}, expected {ep_model_dim}"
            )

    def test_usage_counts_sum_matches_top_k(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        activations = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, indices = router(activations)
        all_experts = list(range(ep_num_experts))
        _, usage = route_tokens(probs, indices, ep_num_experts, all_experts, activations)
        total = sum(usage.values())
        assert total == batch_size * small_seq_len * ep_top_k


# ---------------------------------------------------------------------------
# compute_expert_contributions
# ---------------------------------------------------------------------------

class TestComputeExpertContributions:

    def _setup(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        expert_dim = ep_model_dim // ep_num_experts
        router = Router(embeddings_dims=ep_model_dim, num_experts=ep_num_experts, top_k=ep_top_k, device=device, noisy_topk=False)
        activations = torch.randn(batch_size, small_seq_len, ep_model_dim)
        probs, indices = router(activations)
        return expert_dim, probs, indices, activations

    def test_output_shape(self, ep_model_dim, ep_num_experts, ep_top_k, ep_num_nodes, device, batch_size, small_seq_len):
        expert_dim, probs, indices, activations = self._setup(ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len)
        shard = get_expert_per_node(0, ep_num_nodes, ep_num_experts)
        local_experts = torch.nn.ModuleDict({str(i): ExpertBlock(expert_dim, device) for i in shard})
        routed, _ = route_tokens(probs, indices, ep_num_experts, shard, activations)
        contribution = compute_expert_contributions(
            peer_tokens=routed,
            expert_probs=probs,
            expert_indices=indices,
            expert_shard_indices=shard,
            local_experts=local_experts,
            expert_dim=expert_dim,
            embedding_dims=ep_model_dim,
            batch_size=batch_size,
            seq_len=small_seq_len,
            device=device,
        )
        assert contribution.shape == (batch_size, small_seq_len, ep_model_dim)

    def test_empty_shard_returns_zeros(self, ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len):
        """When no experts are assigned tokens, contribution tensor is all zeros."""
        expert_dim, probs, indices, activations = self._setup(ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len)
        local_experts = torch.nn.ModuleDict({})
        contribution = compute_expert_contributions(
            peer_tokens={},
            expert_probs=probs,
            expert_indices=indices,
            expert_shard_indices=[],
            local_experts=local_experts,
            expert_dim=expert_dim,
            embedding_dims=ep_model_dim,
            batch_size=batch_size,
            seq_len=small_seq_len,
            device=device,
        )
        assert torch.all(contribution == 0)

    def test_gradient_flows_through_experts(self, ep_model_dim, ep_num_experts, ep_top_k, ep_num_nodes, device, batch_size, small_seq_len):
        expert_dim, probs, indices, activations = self._setup(ep_model_dim, ep_num_experts, ep_top_k, device, batch_size, small_seq_len)
        shard = get_expert_per_node(0, ep_num_nodes, ep_num_experts)
        local_experts = torch.nn.ModuleDict({str(i): ExpertBlock(expert_dim, device) for i in shard})
        routed, _ = route_tokens(probs, indices, ep_num_experts, shard, activations)
        contribution = compute_expert_contributions(
            peer_tokens=routed,
            expert_probs=probs,
            expert_indices=indices,
            expert_shard_indices=shard,
            local_experts=local_experts,
            expert_dim=expert_dim,
            embedding_dims=ep_model_dim,
            batch_size=batch_size,
            seq_len=small_seq_len,
            device=device,
        )
        contribution.sum().backward()
        params_with_grad = [p for p in local_experts.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0


# ---------------------------------------------------------------------------
# End-to-end EP pipeline
# ---------------------------------------------------------------------------

class TestEPPipeline:
    """Simulate the full distributed EP forward pass in a single process."""

    def _make_mixtral(self, vocab_size, model_dim, num_heads, num_layers, device):
        return Mixtral(
            vocab_size=vocab_size, embeddings_dims=model_dim, no_of_heads=num_heads,
            no_of_decoder_layers=num_layers, device=device,
        )

    def _run_pipeline(self, num_nodes, num_experts, model_dim, num_heads, top_k,
                      vocab_size, batch_size, seq_len, num_layers, device):
        assert num_experts % num_nodes == 0
        expert_dim = model_dim // num_experts

        text_embeddings = TextEmbeddings(vocab_size=vocab_size, embeddings_dims=model_dim, device=device)
        router = Router(embeddings_dims=model_dim, num_experts=num_experts, top_k=top_k, device=device, noisy_topk=False)

        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        activations = text_embeddings(tokens)
        expert_probs, expert_indices = router(activations)

        model_cfg = dict(top_k=top_k, device=device, noisy_topk=False)
        aggregated = torch.zeros(batch_size, seq_len, model_dim, device=device)

        for worker_rank in range(num_nodes):
            shard = get_expert_per_node(worker_rank, num_nodes, num_experts)
            mixtral_for_rank = self._make_mixtral(vocab_size, model_dim, num_heads, num_layers, device)
            _, ep_layers = get_model_per_node(
                model=mixtral_for_rank, num_nodes=num_nodes, local_rank=worker_rank,
                total_layers=num_layers, model_type="causal_mixtral",
                num_experts=num_experts, model_config=model_cfg,
            )
            local_experts = nn.ModuleDict({
                str(idx): ep_layers[f"model.expert_{idx}"] for idx in shard
            })
            routed, _ = route_tokens(expert_probs, expert_indices, num_experts, shard, activations)
            contribution = compute_expert_contributions(
                peer_tokens=routed,
                expert_probs=expert_probs,
                expert_indices=expert_indices,
                expert_shard_indices=shard,
                local_experts=local_experts,
                expert_dim=expert_dim,
                embedding_dims=model_dim,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            aggregated += contribution

        # Last rank Mixtral — build one more for the last rank to run the transformer
        last_mixtral = self._make_mixtral(vocab_size, model_dim, num_heads, num_layers, device)
        _, last_ep_layers = get_model_per_node(
            model=last_mixtral, num_nodes=num_nodes, local_rank=num_nodes - 1,
            total_layers=num_layers, model_type="causal_mixtral",
            num_experts=num_experts, model_config=model_cfg,
        )
        mixtral = last_ep_layers["model.mixtral"]
        logits = mixtral(aggregated)
        return logits, aggregated

    def test_output_shape(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
                          small_vocab_size, batch_size, small_seq_len, num_layers, device):
        logits, _ = self._run_pipeline(
            ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
            small_vocab_size, batch_size, small_seq_len, num_layers, device,
        )
        assert logits.shape == (batch_size, small_seq_len, small_vocab_size)

    def test_loss_and_backward(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
                               small_vocab_size, batch_size, small_seq_len, num_layers, device):
        logits, _ = self._run_pipeline(
            ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
            small_vocab_size, batch_size, small_seq_len, num_layers, device,
        )
        target = torch.randint(0, small_vocab_size, (batch_size, small_seq_len))
        B, T, C = logits.shape
        loss = nn.functional.cross_entropy(logits.view(B * T, C), target.view(B * T))
        loss.backward()
        assert loss.item() > 0

    def test_aggregated_shape(self, ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
                              small_vocab_size, batch_size, small_seq_len, num_layers, device):
        _, aggregated = self._run_pipeline(
            ep_num_nodes, ep_num_experts, ep_model_dim, ep_num_heads, ep_top_k,
            small_vocab_size, batch_size, small_seq_len, num_layers, device,
        )
        assert aggregated.shape == (batch_size, small_seq_len, ep_model_dim)

    def test_six_experts_three_nodes(self, ep_num_heads, ep_top_k, small_vocab_size, batch_size, small_seq_len, num_layers, device):
        """6 experts / 3 nodes = 2 each. Uses model_dim=96 (96//6=16 expert_dim, 96//4=24 head_dim even)."""
        logits, _ = self._run_pipeline(
            num_nodes=3, num_experts=6, model_dim=96, num_heads=ep_num_heads, top_k=ep_top_k,
            vocab_size=small_vocab_size, batch_size=batch_size, seq_len=small_seq_len,
            num_layers=num_layers, device=device,
        )
        assert logits.shape == (batch_size, small_seq_len, small_vocab_size)

    def test_single_node_full_pipeline(self, ep_num_experts, ep_top_k, ep_model_dim, ep_num_heads,
                                       small_vocab_size, batch_size, small_seq_len, num_layers, device):
        """1 node handles all components."""
        logits, _ = self._run_pipeline(
            num_nodes=1, num_experts=ep_num_experts, model_dim=ep_model_dim, num_heads=ep_num_heads,
            top_k=ep_top_k, vocab_size=small_vocab_size, batch_size=batch_size,
            seq_len=small_seq_len, num_layers=num_layers, device=device,
        )
        assert logits.shape == (batch_size, small_seq_len, small_vocab_size)
