"""Freeze a rejected K-winner throughput characterization, not model selection."""
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = ROOT / "output/optimization/factory-round3/f2-rejected-id1-calibration-capacity-plan-before-timing.json"
    plan = copy.deepcopy(json.loads(prior.read_text(encoding="utf-8")))
    frozen = ROOT / "output/optimization/factory-round3/winner-c200-density-freeze-before-comparison.json"
    if sha(frozen) != "030281c7b8da5d98f2b84362c7a5059b13ffaeaf7daf9eaae3addee112cae070":
        raise ValueError("K winner freeze changed")
    winner = json.loads(frozen.read_text(encoding="utf-8"))["winner"]
    path = Path(winner["model_path"])
    if (winner["candidate_id"] != 3 or sha(path) != winner["model_sha256"]
            or sha(path.with_suffix(".onnx.json")) != winner["metadata_sha256"]):
        raise ValueError("Selected K model identity changed")
    old_stem = "spatial_round3i_normal_density_f16-step1"
    new_stem = path.stem
    # Carry over the exact established workload structure, replacing model and
    # destination strings only. Scientific pins below are rebuilt, never reused.
    def replace(value):
        if isinstance(value, dict):
            return {replace(k): replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, str):
            return value.replace(old_stem, new_stem).replace("factory-round3i-density-capacity", "factory-round3k-capacity").replace("failed-density-id1", "failed-c200-density-id3")
        return value
    plan = replace(plan)
    plan["written_utc"] = datetime.now(timezone.utc).isoformat()
    plan["purpose"] = "Capacity characterization of rejected K candidate3, the already globally frozen winner. No alternative candidate/point, original comparison NN or adoption."
    plan["prior_workload_template_sha256"] = sha(prior)
    plan["selected_winner_freeze_sha256"] = sha(frozen)
    plan.pop("f2_rejected_freeze_sha256", None)
    plan["runtime_script_sha256"] = sha(ROOT / "scripts/benchmark_runtime.py")
    plan["reporting"] = "Keep rejected K status irrespective of capacity; retain complete capture/processed/drop/late/source-skip/errors and all stage/queue tails. Public still replay does not establish physical production-line coverage."
    for key, value in list(plan.items()):
        if key not in {"model_records"}:
            continue
        for name, record in value.items():
            candidate = Path(record["path"])
            metadata_path = candidate.with_suffix(candidate.suffix + ".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record.update(sha256=sha(candidate), metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path),
                          policy={k: metadata[k] for k in record["policy"]}, split_items=metadata["split_items"])
    # Rebind every path-keyed source pin to current executed source, including
    # optional precision hooks whose default remains unchanged for these models.
    def refresh(value):
        if isinstance(value, dict):
            for key, item in value.items():
                target = ROOT / key
                if isinstance(item, str) and len(item) == 64 and target.is_file():
                    value[key] = sha(target)
                else:
                    refresh(item)
        elif isinstance(value, list):
            for item in value:
                refresh(item)
    refresh(plan)
    plan["actual_runtime_source_sha256"] = {name: sha(ROOT / name) for name in [
        "edge_runtime/profiles/factory.py", "edge_runtime/profiles/factory_defect.py", "edge_runtime/profiles/factory_encoder.py",
        "edge_runtime/runtime.py", "scripts/benchmark_factory_accuracy.py", "scripts/benchmark_runtime.py"]}
    target = ROOT / "output/optimization/factory-round3/k-rejected-winner3-calibration-capacity-plan-before-timing.json"
    if target.exists():
        raise ValueError("Preserve previous K timing declaration")
    target.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(target, sha(target))


if __name__ == "__main__":
    main()
