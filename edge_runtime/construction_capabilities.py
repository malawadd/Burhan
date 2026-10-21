"""Separate visible-damage model availability from inspection-plan coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPABILITIES_PATH = ROOT / "config" / "construction-capabilities.json"


def registered_models():
    from .profiles import construction_model as models

    result = {mode: spec for mode, spec in models.MODELS.items()}
    for mode, attribute in (("crack", "CRACK_VARIANTS"), ("damage", "DAMAGE_VARIANTS")):
        result.update({f"{mode}:{name}": spec
                       for name, spec in getattr(models, attribute, {}).items()})
    return result


def load_capabilities(path=None, *, model_specs=None, root=None):
    """A catalog entry alone must never claim an installed detector."""
    target = Path(path) if path else CAPABILITIES_PATH
    if not target.is_file():
        return {"schema_version": 1, "available": False, "categories": [],
                "scope": "Visible damage capabilities have not been registered"}
    data = json.loads(target.read_text(encoding="utf-8"))
    specs = registered_models() if model_specs is None else model_specs
    model_root = (Path(root) if root else ROOT) / "models" / "construction"
    verified = {}
    def installed_model(reference):
        if reference not in verified:
            installed = False
            spec = specs.get(reference)
            if spec:
                asset = (model_root / spec["file"]).resolve()
                installed = (asset.is_relative_to(model_root.resolve()) and asset.is_file()
                             and hashlib.sha256(asset.read_bytes()).hexdigest() == spec["sha256"])
                if installed and spec.get("decoder") == "tile_patch_bank":
                    try:
                        metadata = json.loads(asset.read_text(encoding="utf-8"))
                        for field in ("encoder", "bank"):
                            dependency = (asset.parent / metadata[field]["file"]).resolve()
                            if (not dependency.is_relative_to(model_root.resolve()) or not dependency.is_file()
                                    or hashlib.sha256(dependency.read_bytes()).hexdigest() != metadata[field]["sha256"]):
                                installed = False
                    except (OSError, ValueError, KeyError, TypeError):
                        installed = False
            verified[reference] = installed
        return verified[reference]

    result = []
    for row in data["categories"]:
        row = dict(row)
        reference = row.get("model_ref")
        reference = reference if isinstance(reference, str) else None
        spec = specs.get(reference)
        row["model_installed"] = installed_model(reference)
        row["registered_capability"] = bool(spec and row["id"] in spec.get("capabilities", []))
        proposed = row.get("model_status", "research_only")
        installed = row["model_installed"] and row["registered_capability"]
        row["effective_status"] = (proposed if proposed in {"supported_experimental", "measured_weak"} and installed else
                                   "unavailable" if proposed in {"supported_experimental", "measured_weak"} else
                                   "unsupported" if proposed == "unsupported" else "research_only")
        if spec and spec.get("diagnostic_only") and proposed != "unsupported":
            row["effective_status"] = "research_only"
        result.append(row)
    names = {"evcivil_yolov8n_round3": "Field video · cracks and spalling",
             "dacl_segformer_b1": "Photo review · multiple conditions (slow)",
             "dacl_segformer_b1_320": "Concrete surfaces · 320px speed diagnostic"}
    models = [{"model_ref": reference, "variant": reference.split(":", 1)[1],
               "label": spec.get("display_name", names.get(reference.split(":", 1)[1], reference)),
               "installed": installed_model(reference), "capabilities": spec["capabilities"],
               "condition_classes": {capability: label for label, capability in spec.get("class_capabilities", {}).items()},
               "input_size": spec.get("input_size"), "diagnostic_only": bool(spec.get("diagnostic_only",False)),
               "validation_scope":spec.get("validation_scope")}
              for reference, spec in specs.items() if reference.startswith("damage:")]
    models.sort(key=lambda row: row["diagnostic_only"])
    return {**data, "available": True, "categories": result, "models": models,
            "code_compliance": "not_determined"}
