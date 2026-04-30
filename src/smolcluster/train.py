"""
Training with EDP, SyncPS, or ModelParallelism for GPT

This script provides both server and worker entry points for distributed
GPT training using EDP, SyncPS, or ModelParallelism algorithmsr: python train.py server <hostname> --algorithm <edp|syncps|mp>
    Worker: python train.py worker <rank> <hostname> --algorithm <edp|syncps|mp>

Examples:
    python train.py server mini1 --algorithm edp
    python train.py worker 1 mini2 --algorithm syncps
    python train.py server mini1 --algorithm mp
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torchinfo
import wandb
import yaml

from smolcluster.algorithms.EDP.server import run_edp_server
from smolcluster.algorithms.EDP.worker import run_edp_worker
from smolcluster.algorithms.ModelParallelism.server import run_modelparallelism_server
from smolcluster.algorithms.ModelParallelism.worker import run_modelparallelism_worker
from smolcluster.algorithms.ModelParallelismPipeline.worker import (
    run_modelparallelism_pipeline_worker,
)
from smolcluster.algorithms.DataParallelism.ClassicDP.worker import run_classicdp_worker
from smolcluster.algorithms.DataParallelism.SynchronousPS.server import run_syncps_server
from smolcluster.algorithms.DataParallelism.SynchronousPS.worker import run_syncps_worker
from smolcluster.algorithms.FSDP.worker_stage0 import run_fsdp_worker as run_fsdp_worker_stage0
from smolcluster.algorithms.FSDP.worker_stage1 import run_fsdp_worker as run_fsdp_worker_stage1
from smolcluster.algorithms.FSDP.worker_stage2 import run_fsdp_worker as run_fsdp_worker_stage2
from smolcluster.algorithms.ExpertParallelism.worker import run_ep_worker
from smolcluster.data.prepare_dataset import prepare_dataset
from smolcluster.models.gpt import BaseTransformer
from smolcluster.models.moe import Mixtral
from smolcluster.utils.device import get_device
from smolcluster.utils.layers import get_model_per_node

# -----------------------------------------------------------------------------
# Configuration and Data Loading
# -----------------------------------------------------------------------------

# Populated by run_discover(); merged into every load_configs() call so that
# run_worker (which re-calls load_configs internally) sees the live IPs.
_discovered_config: dict = {}


def load_configs(algorithm: str = "syncps"):
    """Load configuration files.

    Args:
        algorithm: Training algorithm to determine which configs to load
    """
    CONFIG_DIR = Path(__file__).parent / "configs"

    # Load appropriate model config based on algorithm
    if algorithm == 'ep':
        # Expert Parallelism uses MoE config
        with open(CONFIG_DIR / "moe_config.yaml") as f:
            model_config = yaml.safe_load(f)
    else:
        # Other algorithms use GPT config
        with open(CONFIG_DIR / "gpt_config.yaml") as f:
            model_config = yaml.safe_load(f)

    # Load appropriate cluster config based on algorithm
    config_file = f"cluster_config_{algorithm}.yaml"
    with open(CONFIG_DIR / config_file) as f:
        cluster_config = yaml.safe_load(f)

    # Overlay any IPs discovered at runtime (set by run_discover)
    if _discovered_config:
        cluster_config.update(_discovered_config)

    return model_config, cluster_config


def load_data(config, world_size: int, seed: int, rank: int, batch_size: int):
    """dataset for the given rank with specified batch size."""

    # Override config batch_size with the provided value
    config_with_batch = config.copy()
    config_with_batch["batch_size"] = batch_size

    train_loader, val_loader, vocab_size, pad_token_id = prepare_dataset(
        config_with_batch, world_size, seed, rank
    )

    return train_loader, val_loader, vocab_size, pad_token_id


def setup_wandb():
    """Setup Weights & Biases authentication."""
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        logger = logging.getLogger("[INIT]")
        logger.info("✅ Logged into wandb using WANDB_API_KEY")
    else:
        logger = logging.getLogger("[INIT]")
        logger.warning("⚠️  WANDB_API_KEY not set - wandb may prompt for login")


def run_server(
    hostname: str, algorithm: str = "syncps", resume_checkpoint_path: str = None
):
    """Run server for GPT training.

    Args:
        hostname: Server hostname
        algorithm: Either 'edp', 'syncps', or 'mp'
        resume_checkpoint_path: Path to checkpoint to resume from
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("[SERVER-MAIN]")

    setup_wandb()

    # Load configs
    gpt_config, cluster_config = load_configs(algorithm)

    # Validate batch_size configuration for EDP
    if algorithm == "edp":
        if "batch_size" not in cluster_config or not isinstance(
            cluster_config["batch_size"], dict
        ):
            logger.error(
                "❌ FATAL: cluster_config_edp.yaml must have 'batch_size' as a dict mapping hostnames to batch sizes"
            )
            sys.exit(1)

        if hostname not in cluster_config["batch_size"]:
            logger.error(
                f"❌ FATAL: No batch_size configured for hostname '{hostname}' in cluster_config_edp.yaml"
            )
            sys.exit(1)

        server_batch_size = cluster_config["batch_size"][hostname]
        logger.info(f"✅ Server batch size: {server_batch_size}")
    else:
        # SyncPS uses global batch_size from gpt_config
        server_batch_size = gpt_config["batch_size"]

    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    world_size = num_workers + 1
    rank = 0  # Server is rank 0

    # Load data with server's batch size
    logger.info(f"Loading {gpt_config.get('dataset_name', 'dataset')} dataset...")
    train_loader, val_loader, vocab_size, pad_token_id = load_data(
        gpt_config, world_size, seed, rank, server_batch_size
    )
    logger.info(
        f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}"
    )

    # Create model
    model = BaseTransformer(
        vocab_size=vocab_size,
        max_seq_len=gpt_config["max_seq_len"],
        model_dim=gpt_config["model_dim"],
        num_layers=gpt_config["num_layers"],
        num_heads=gpt_config["num_heads"],
        ff_dim=gpt_config["ff_dim"],
        dropout=gpt_config["dropout"],
    )
    device = get_device()
    model = model.to(device)
    logger.info(f"Model initialized on device: {device}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=gpt_config["learning_rate"],
        weight_decay=gpt_config["weight_decay"],
    )

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-server-{hostname}_lr{gpt_config['learning_rate']}_bs{server_batch_size}_workers{num_workers}",
        config={
            **gpt_config,
            "server_hostname": hostname,
            "num_workers": num_workers,
            "algorithm": algorithm,
            "server_batch_size": server_batch_size,
        },
    )

    # Run server with selected algorithm
    if algorithm == "mp_pipeline":
        logger.error(
            "❌ FATAL: mp_pipeline algorithm does not use a server. Only launch workers."
        )
        sys.exit(1)
    elif algorithm == "classicdp":
        logger.error(
            "❌ FATAL: classicdp algorithm does not use a server. Only launch workers."
        )
        sys.exit(1)
    elif algorithm == "fsdp":
        logger.error(
            "❌ FATAL: fsdp algorithm does not use a server. Only launch workers."
        )
        sys.exit(1)
    
    logger.info(f"Starting {algo_name} server...")
    if algorithm == "edp":
        run_edp_server(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            hostname=hostname,
            device=device,
            criterion=criterion,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    elif algorithm == "mp":
        run_modelparallelism_server(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            hostname=hostname,
            device=device,
            criterion=criterion,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    else:  # syncps
        run_syncps_server(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            hostname=hostname,
            device=device,
            criterion=criterion,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    wandb.finish()


def run_worker(
    worker_rank: int,
    hostname: str,
    algorithm: str = "syncps",
    resume_checkpoint_path: str = None,
):
    """Run worker for GPT training.

    Args:
        worker_rank: Worker rank (1-indexed)
        hostname: Worker hostname
        algorithm: Either 'edp', 'syncps', or 'mp'
        resume_checkpoint_path: Path to checkpoint to resume from
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Worker rank is 0-indexed internally but 1-indexed in command-line
    local_rank = worker_rank
    logger = logging.getLogger(f"[WORKER-{local_rank}-MAIN]")

    setup_wandb()

    # Load configs
    gpt_config, cluster_config = load_configs(algorithm)

    # Validate batch_size configuration for EDP
    if algorithm == "edp":
        if "batch_size" not in cluster_config or not isinstance(
            cluster_config["batch_size"], dict
        ):
            logger.error(
                "❌ FATAL: cluster_config_edp.yaml must have 'batch_size' as a dict mapping hostnames to batch sizes"
            )
            sys.exit(1)

        if hostname not in cluster_config["batch_size"]:
            logger.error(
                f"❌ FATAL: No batch_size configured for hostname '{hostname}' in cluster_config_edp.yaml"
            )
            sys.exit(1)

        worker_batch_size = cluster_config["batch_size"][hostname]
        logger.info(f"✅ Worker batch size: {worker_batch_size}")
    else:
        # SyncPS uses global batch_size from gpt_config
        worker_batch_size = gpt_config["batch_size"]

    # Setup parameters
    num_workers = cluster_config["num_workers"]
    seed = cluster_config.get("seed", 42)
    # For mp_pipeline, classicdp, fsdp, and ep: world_size is just num_workers (no separate server)
    # For other algorithms, world_size includes server (num_workers + 1)
    world_size = num_workers if algorithm in ["mp_pipeline", "classicdp", "fsdp", "ep"] else num_workers + 1

    # Get server connection info (only needed for algorithms with server)
    host_ip = None
    if algorithm in ["mp_pipeline", "classicdp", "fsdp", "ep"]:
        # Pipeline topology, ClassicDP, FSDP, and EP don't use host_ip for server connection
        pass
    elif algorithm == "syncps":
        workers_cfg = cluster_config.get("workers", [])
        if not isinstance(workers_cfg, list) or len(workers_cfg) == 0:
            raise ValueError(
                "SyncPS requires cluster_config_syncps.yaml workers[] with hostname/rank/ip entries"
            )

        worker_entry = next(
            (
                w
                for w in workers_cfg
                if w.get("hostname") == hostname and int(w.get("rank", -1)) == int(local_rank)
            ),
            None,
        )
        if worker_entry is None:
            raise ValueError(
                f"SyncPS worker entry not found for hostname='{hostname}', rank={local_rank}. "
                "Define it under workers[] in cluster_config_syncps.yaml"
            )

        worker_ip = str(worker_entry.get("ip", "")).strip()
        if not worker_ip:
            raise ValueError(
                f"SyncPS worker '{hostname}' rank {local_rank} is missing workers[].ip in cluster_config_syncps.yaml"
            )

        server_hostname = cluster_config["server"]
        server_ip = str(cluster_config.get("host_ip", {}).get(server_hostname, "")).strip()
        if not server_ip:
            raise ValueError(
                f"SyncPS server '{server_hostname}' is missing host_ip mapping in cluster_config_syncps.yaml"
            )

        host_ip = server_ip
        logger.info(
            f"SyncPS mapping -> worker={hostname} rank={local_rank} worker_ip={worker_ip} server={server_hostname} server_ip={server_ip}"
        )
    else:
        # Require host_ip for other algorithms
        if algorithm == "mp" or algorithm == "mp_pipeline":
            # Workers connect TO the server — look up the server's IP, not our own.
            server_hostname = cluster_config.get("server", "")
            host_ip = cluster_config.get("host_ip", {}).get(server_hostname, "")
            if not host_ip:
                raise ValueError(
                    f"{algorithm.upper()}: server '{server_hostname}' missing from host_ip in config"
                )
        else:
            host_ip = cluster_config["host_ip"][hostname]
    
    port_config = cluster_config["port"]
    if isinstance(port_config, dict):
        # For mp_pipeline, classicdp, fsdp, and ep: get worker rank 0's hostname; for others, use server
        if algorithm in ["mp_pipeline", "classicdp", "fsdp", "ep"]:
            # For different topologies based on algorithm
            topology_key = "pipelineTopology" if algorithm == "mp_pipeline" else "allToAllTopology"
            workers_list = cluster_config[topology_key]["workers"]["regular"]
            coordinator_hostname = next(
                w["hostname"] for w in workers_list if w["rank"] == 0
            )
        else:
            coordinator_hostname = cluster_config["server"]
        port = port_config.get(coordinator_hostname, port_config.get("default", 65432))
    else:
        port = port_config

    # Load data with worker's batch size
    logger.info(f"Loading {gpt_config.get('dataset_name', 'dataset')} dataset...")
    train_loader, val_loader, vocab_size, pad_token_id = load_data(
        gpt_config, world_size, seed, local_rank, worker_batch_size
    )
    logger.info(
        f"Data ready. Train size: {len(train_loader)}, Val size: {len(val_loader)}"
    )

    # Get device before model creation
    device = get_device()

    # Create model
    if algorithm == 'ep':
        # EP: create the Mixtral transformer (last rank will use it; get_model_per_node handles assignment)
        model = Mixtral(
            vocab_size=vocab_size,
            embeddings_dims=gpt_config["embedding_dims"],
            no_of_heads=gpt_config.get("no_of_heads", gpt_config.get("num_heads", 12)),
            no_of_decoder_layers=gpt_config["num_layers"],
            device=device,
            attn_dropout=gpt_config.get("attn_dropout", 0.1),
            dropout=gpt_config.get("dropout", 0.1),
        )
    else:
        model = BaseTransformer(
            vocab_size=vocab_size,
            max_seq_len=gpt_config["max_seq_len"],
            model_dim=gpt_config["model_dim"],
            num_layers=gpt_config["num_layers"],
            num_heads=gpt_config["num_heads"],
            ff_dim=gpt_config["ff_dim"],
            dropout=gpt_config["dropout"],
        )
   
    
    # For FSDP Stage 3 and EP, skip moving full model to device
    # Model sharding happens later on CPU or per-worker
    if algorithm not in ['fsdp', 'ep']:
        
        model = model.to(device)
        logger.info(f"Model initialized on device: {device}")

        # Print model summary
        logger.info("Model Summary:")
        summary = torchinfo.summary(
            model,
            input_size=(worker_batch_size, gpt_config["max_seq_len"]),
            device=device,
            dtypes=[torch.long],
        )
        logger.info(f"\n{summary}")
    else:
        logger.info(f"{algorithm.upper()}: Model will be sharded, skipping full model device transfer")

    # Create optimizer (not needed for FSDP Stage 2 or EP, handled per-worker)
    optimizer = None
    if algorithm not in ['fsdp', 'ep']:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=gpt_config["learning_rate"],
            weight_decay=gpt_config["weight_decay"],
        )

    # Create criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # Initialize W&B
    algo_name = algorithm.upper()
    wandb.init(
        project="smolcluster",
        name=f"{algo_name}-worker-{hostname}_rank{local_rank}_lr{gpt_config['learning_rate']}_bs{worker_batch_size}",
        config={
            **gpt_config,
            "worker_rank": local_rank,
            "worker_hostname": hostname,
            "algorithm": algorithm,
            "worker_batch_size": worker_batch_size,
        },
    )

    # Run worker with selected algorithm
    logger.info(f"Starting {algo_name} worker {local_rank}...")
    if algorithm == "edp":
        run_edp_worker(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    elif algorithm == "mp":
        run_modelparallelism_worker(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    elif algorithm == "mp_pipeline":
        run_modelparallelism_pipeline_worker(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    elif algorithm == 'syncps':  # syncps
            run_syncps_worker(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=gpt_config,
                worker_rank=local_rank,
                hostname=hostname,
                device=device,
                criterion=criterion,
                host_ip=host_ip,
                port=port,
                resume_checkpoint_path=resume_checkpoint_path,
            )
    elif algorithm == 'classicdp':
        run_classicdp_worker(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=gpt_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    elif algorithm == 'fsdp':
        # Select FSDP stage: 0 = ZeRO-0 (optimizer), 1 = ZeRO-1 (+gradient), 2 = ZeRO-2 (+parameter)
        fsdp_stage = cluster_config.get('fsdp_stage', 0)
        if fsdp_stage == 0:
            run_fsdp_worker_stage0(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=gpt_config,
                cluster_config=cluster_config,
                worker_rank=local_rank,
                hostname=hostname,
                device=device,
                criterion=criterion,
                host_ip=host_ip,
                port=port,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        elif fsdp_stage == 1:
            run_fsdp_worker_stage1(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=gpt_config,
                cluster_config=cluster_config,
                worker_rank=local_rank,
                hostname=hostname,
                device=device,
                criterion=criterion,
                host_ip=host_ip,
                port=port,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        elif fsdp_stage == 2:
            # FSDP Stage 3: Shard model BEFORE passing to worker (never load full model on worker GPU)
            logger.info(f"Sharding model for FSDP Stage 3 worker rank {local_rank}...")
            num_nodes = cluster_config["num_nodes"]
            num_layers = cluster_config["num_layers"]
            
            # Extract this worker's parameter shard on CPU
            _, out_layers = get_model_per_node(
                model=model,
                num_nodes=num_nodes,
                local_rank=local_rank,
                total_layers=num_layers,
            )
            
            # Extract owned parameters as dict (layer_name -> parameter tensor)
            owned_params_dict = {}
            for layer_name, module in out_layers.items():
                for param_name, param in module.named_parameters():
                    full_param_name = f"{layer_name}.{param_name}"
                    owned_params_dict[full_param_name] = param.data.cpu().clone()
            
            logger.info(f"Worker rank {local_rank} will own {len(owned_params_dict)} parameters")
            
            # Create empty skeleton from full model (structure only, no weights)
            model_skeleton = model
            with torch.no_grad():
                for param in model_skeleton.parameters():
                    param.data = torch.empty(0, device='cpu')
            
            # Delete full model and shards to free memory
            del model, out_layers
            import gc
            gc.collect()
            
            # Pass empty skeleton and shard dict to worker (no full model weights!)
            run_fsdp_worker_stage2(
                model_skeleton=model_skeleton,
                owned_params_dict=owned_params_dict,
                train_loader=train_loader,
                val_loader=val_loader,
                config=gpt_config,
                cluster_config=cluster_config,
                worker_rank=local_rank,
                hostname=hostname,
                device=device,
                criterion=criterion,
                host_ip=host_ip,
                port=port,
                resume_checkpoint_path=resume_checkpoint_path,
            )
        else:
            raise ValueError(f"Invalid fsdp_stage: {fsdp_stage}. Must be 0, 1, or 2.")
    elif algorithm == 'ep':
        # Expert Parallelism: Each worker processes tokens for its assigned experts
        run_ep_worker(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader if local_rank == 0 else None,  # Only rank 0 needs val loader
            config=gpt_config,
            cluster_config=cluster_config,
            worker_rank=local_rank,
            hostname=hostname,
            device=device,
            criterion=criterion,
            host_ip=host_ip,
            port=port,
            resume_checkpoint_path=resume_checkpoint_path,
        )
    wandb.finish()


def run_discover(algorithm: str, cluster: str, world_size: int, resume_checkpoint_path: str = None):
    """Discover peers via grove mDNS/AWDL, patch cluster_config with real IPs, then train.

    Activated only when running through grove (grove start/join).
    Existing server/worker paths are untouched.
    """
    global _discovered_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    from smolcluster.discovery import discover

    logger = logging.getLogger("[DISCOVER]")
    logger.info(f"Discovering {world_size} peers for cluster '{cluster}'...")

    my_rank, peers, zc = discover(cluster, world_size, timeout=120.0)
    hostname = peers[my_rank]["hostname"]
    logger.info(f"Rank {my_rank} / {world_size} — peers: { {r: p['host'] for r, p in peers.items()} }")

    # Build the discovered config overlay — run_worker's load_configs will merge this in
    if algorithm == "classicdp":
        base_port = int(os.environ.get("SMOLCLUSTER_PORT", "65432"))
        workers = [
            {"hostname": p["hostname"], "rank": r, "ip": p["host"], "port": base_port + r}
            for r, p in peers.items()
        ]
        _discovered_config = {
            "num_workers": world_size,
            "allToAllTopology": {"workers": {"regular": workers, "tablets": []}},
            "port": {p["hostname"]: base_port + r for r, p in peers.items()},
        }

    # extend here for other algos:
    # elif algorithm == "syncps":
    #     _discovered_config = {"host_ip": {peers[0]["hostname"]: peers[0]["host"]}}
    # elif algorithm == "fsdp": ...

    try:
        run_worker(my_rank, hostname, algorithm, resume_checkpoint_path)
    finally:
        zc.close()


def main():
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description="Distributed GPT Training")
    parser.add_argument(
        "mode", choices=["server", "worker", "dashboard", "grove", "discover"],
        help="Run as server, worker, dashboard, grove, or discover (auto-discovery via grove)"
    )
    parser.add_argument("arg1", help="Hostname (server mode) or rank (worker mode)")
    parser.add_argument("arg2", nargs="?", help="Hostname (worker mode only)")
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["edp", "syncps", "mp", "mp_pipeline", "classicdp", "fsdp", "ep"],
        default="syncps",
        help="Training algorithm to use (default: syncps)",
    )
    parser.add_argument(
        "-r",
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Dashboard shortcut — no further args needed
    if len(sys.argv) >= 2 and sys.argv[1] == "dashboard":
        from dashboard.__main__ import main as dashboard_main
        import sys as _sys
        _sys.argv = [_sys.argv[0]] + _sys.argv[2:]  # strip "dashboard" so argparse in __main__ works
        dashboard_main()
        return

    # Discover mode: grove start/join passes argv like ['train.py', 'discover', '-a', 'classicdp']
    if len(sys.argv) >= 2 and sys.argv[1] == "discover":
        alg_parser = argparse.ArgumentParser(add_help=False)
        alg_parser.add_argument("-a", "--algorithm",
                                 default=os.environ.get("SMOLCLUSTER_ALGO", "syncps"))
        alg_parser.add_argument("-r", "--resume-checkpoint", default=None)
        alg_args, _ = alg_parser.parse_known_args(sys.argv[2:])
        cluster = os.environ.get("SMOLCLUSTER_CLUSTER", "smolcluster-run")
        world_size = int(os.environ.get("SMOLCLUSTER_WORLD_SIZE", "2"))
        run_discover(alg_args.algorithm, cluster, world_size, alg_args.resume_checkpoint)
        return

    # Handle both new argparse format and legacy positional format
    if len(sys.argv) >= 2 and sys.argv[1] in ["server", "worker"]:
        # Try to parse with argparse first
        if (
            "--algorithm" in sys.argv
            or "-a" in sys.argv
            or "--resume-checkpoint" in sys.argv
            or "-r" in sys.argv
        ):
            args = parser.parse_args()
            mode = args.mode
            algorithm = args.algorithm
            resume_checkpoint_path = args.resume_checkpoint

            # Parse positional args based on mode
            if mode == "server":
                hostname = args.arg1
                worker_rank = None
            else:  # worker
                worker_rank = int(args.arg1)
                hostname = args.arg2
                if hostname is None:
                    print("Error: Worker mode requires both rank and hostname")
                    parser.print_help()
                    sys.exit(1)
        else:
            # Legacy format: mode hostname [rank]
            mode = sys.argv[1]
            resume_checkpoint_path = None
            if len(sys.argv) < 3:
                parser.print_help()
                sys.exit(1)

            if mode == "server":
                hostname = sys.argv[2]
                algorithm = sys.argv[3] if len(sys.argv) > 3 else "syncps"
                worker_rank = None
            else:  # worker
                if len(sys.argv) < 4:
                    parser.print_help()
                    sys.exit(1)
                worker_rank = int(sys.argv[2])
                hostname = sys.argv[3]
                algorithm = sys.argv[4] if len(sys.argv) > 4 else "syncps"
    else:
        parser.print_help()
        sys.exit(1)

    # Validate algorithm
    if algorithm not in ["edp", "syncps", "mp", "mp_pipeline", "classicdp", "fsdp", "ep"]:
        print(
            f"Error: Invalid algorithm '{algorithm}'. Must be 'edp', 'syncps', 'mp', 'mp_pipeline', 'classicdp', 'fsdp', or 'ep'"
        )
        sys.exit(1)

    # Run appropriate mode
    if mode == "server":
        run_server(hostname, algorithm, resume_checkpoint_path)
    elif mode == "worker":
        if worker_rank is None:
            print("Error: Worker mode requires rank argument")
            parser.print_help()
            sys.exit(1)
        run_worker(worker_rank, hostname, algorithm, resume_checkpoint_path)
    elif mode == "grove":
        cluster = os.environ.get("SMOLCLUSTER_CLUSTER", "smolcluster-run")
        world_size = int(os.environ.get("SMOLCLUSTER_WORLD_SIZE", "2"))
        run_discover(algorithm, cluster, world_size, resume_checkpoint_path)


if __name__ == "__main__":
    main()
