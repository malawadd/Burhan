"""Research checklist inventory; observations never imply a code approval."""
import json
from pathlib import Path


CATALOG_PATH = Path(__file__).parent / "data" / "construction-inspections.json"


def load_catalog(path=None, *, verify_models=True):
    target = Path(path) if path else CATALOG_PATH
    if not target.is_file():
        return {"schema_version": 1, "available": False, "stages": [], "inspections": [],
                "scope": "Research catalog unavailable; no construction compliance assessment"}
    result = json.loads(target.read_text(encoding="utf-8"))
    stages = result.get("stages", [])
    rows = result.get("inspections", [])
    stage_ids = [stage["id"] for stage in stages]
    identifiers = [row["id"] for row in rows]
    if len(set(stage_ids)) != len(stage_ids) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Construction catalog identifiers must be unique")
    if any(row.get("stage") not in stage_ids for row in rows):
        raise ValueError("Construction catalog contains an unknown stage")
    result["stages"] = [{**stage, "assessment": "not_assessed"} for stage in stages]
    result["inspections"] = [{**row, "assessment": "not_assessed", "default_assessment": "unassessed",
                              "code_compliance": "not_determined"} for row in rows]
    result["available"] = True
    result["compliance_determined"] = False
    if not verify_models:
        result["visual_models"] = {}
        return result
    try:
        from .profiles.construction_model import ROOT
        from .construction_capabilities import registered_models
        import hashlib
        capabilities = {}
        for mode, spec in registered_models().items():
            model = ROOT / "models" / "construction" / spec["file"]
            installed = model.is_file() and hashlib.sha256(model.read_bytes()).hexdigest() == spec["sha256"]
            capabilities[mode] = {"installed": installed, "capability_ids": spec["capabilities"],
                                  "scope": "Visible candidates only; no Saudi code acceptance validation"}
        result["visual_models"] = capabilities
    except ImportError:
        result["visual_models"] = {}
    return result
