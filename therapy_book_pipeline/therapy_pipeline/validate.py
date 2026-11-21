import json
from pathlib import Path
from pydantic import ValidationError
from .models import Strategy, ScenarioSeed

def validate(output_dir: Path):
    errors, warnings = [], []
    for filename, model in [("strategies.json", Strategy)]:
        path = output_dir / filename
        if not path.exists():
            errors.append(f"Missing {filename}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for i, item in enumerate(data):
                try: model.model_validate(item)
                except ValidationError as e: errors.append(f"{filename}[{i}]: {e}")
            ids = [x.get("strategy_id") for x in data]
            if len(ids) != len(set(ids)): warnings.append(f"{filename}: duplicate IDs detected")
        except Exception as e: errors.append(f"{filename}: invalid JSON: {e}")
    for filename in ("scenarios.json", "scenario_seeds.json", "anxiety_rubric.json"):
        path = output_dir / filename
        if not path.exists():
            errors.append(f"Missing {filename}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if filename == "scenarios.json":
                for i, row in enumerate(data):
                    score = row.get("initial_anxiety_estimate")
                    if not isinstance(score, int) or not 1 <= score <= 10:
                        errors.append(f"{filename}[{i}]: anxiety estimate must be integer 1..10")
                    if row.get("origin") == "synthetic_expansion_from_book_seed" and not row.get("is_synthetic"):
                        errors.append(f"{filename}[{i}]: synthetic origin missing is_synthetic=true")
        except Exception as e:
            errors.append(f"{filename}: invalid JSON: {e}")
    if errors:
        print("VALIDATION FAILED")
        for x in errors: print("ERROR:", x)
    else:
        print("Validation passed.")
    for x in warnings: print("WARNING:", x)
    return not errors
