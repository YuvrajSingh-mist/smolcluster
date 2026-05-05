"""Shared constants for the node manager: remote repo path, log sanitizer, and training algorithm map."""
import re

REMOTE_REPO = "~/smolcluster"

# Strip C0 control chars except ESC so ANSI colors can be rendered in the UI.
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1A\x1C-\x1F\x7F]")


def _sanitize_log_line(line: str) -> str:
    return _CTRL_RE.sub("", line)


# algorithm → (config_filename, script_filename, topology_type)
# topology_type:
#   "flat_server"   — dedicated server + flat workers list
#   "nested_server" — dedicated server + workers.{regular,tablets}
#   "allToAll"      — all nodes in allToAllTopology.workers.regular
#   "pipeline"      — all nodes in pipelineTopology.workers.regular
#   "grpo"          — dedicated server + inference-style workers + total_num_nodes
_TRAINING_ALGO_MAP: dict = {
    "syncps":      ("cluster_config_syncps.yaml",       "launch_syncps_train_gpt.sh",      "flat_server"),
    "mp":          ("cluster_config_mp.yaml",            "launch_mp_train_gpt.sh",          "nested_server"),
    "classicdp":   ("cluster_config_classicdp.yaml",    "launch_dp_train_gpt.sh",          "allToAll"),
    "fsdp":        ("cluster_config_fsdp.yaml",          "launch_fsdp_train_gpt.sh",        "allToAll"),
    "ep":          ("cluster_config_ep.yaml",             "launch_ep_train_moe.sh",          "allToAll"),
    "mp_pipeline": ("cluster_config_mp_pipeline.yaml",  "launch_mp_pipeline_train_gpt.sh", "pipeline"),
    "edp":         ("cluster_config_edp.yaml",           "launch_edp_train_gpt.sh",         "flat_server"),
    "grpo":        (
        "inference/cluster_config_inference.yaml",
        "../../src/smolcluster/applications/reasoning/grpo/scripts/launch_grpo_train.sh",
        "grpo",
    ),
}
