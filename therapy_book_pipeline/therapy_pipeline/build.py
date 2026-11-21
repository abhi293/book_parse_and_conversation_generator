import json, re, hashlib
from pathlib import Path
from rapidfuzz import fuzz
from .models import Strategy, ScenarioSeed

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

def deduplicate_strategies(items, threshold=92):
    groups = []
    for item in items:
        name = re.sub(r"\W+", " ", item.get("strategy_name", "").lower()).strip()
        match = None
        for group in groups:
            other = re.sub(r"\W+", " ", group["strategy_name"].lower()).strip()
            if item.get("modality") == group.get("modality") and fuzz.ratio(name, other) >= threshold:
                match = group
                break
        if match is None:
            groups.append(dict(item))
        else:
            for key in ("source_refs",):
                match[key] = list(dict.fromkeys(match.get(key, []) + item.get(key, [])))
            for key in ("description", "when_to_use", "how_to_apply", "limitations_or_cautions"):
                val = item.get(key, "")
                if val and val not in match.get(key, ""):
                    match[key] = (match.get(key, "") + "\n" + val).strip()
            match["confidence"] = max(match.get("confidence", 0), item.get("confidence", 0))
    return groups

def build_scenarios(seeds, per_seed=3):
    # Expansion creates clearly labeled synthetic variants; it does not claim they occurred in the books.
    contexts = [
        ("university", "A university student facing academic pressure and uncertainty"),
        ("workplace", "An early-career employee dealing with performance expectations"),
        ("relationships", "A person navigating tension in an important relationship"),
        ("transition", "A person adjusting to a major life transition")
    ]
    trajectories = ["gradual_relief", "oscillating", "spike_then_settle"]
    out = []
    for seed in seeds:
        for i in range(per_seed):
            context = contexts[i % len(contexts)]
            trajectory = trajectories[i % len(trajectories)]
            sid = hashlib.sha1(f'{seed.get("seed_id")}-{i}'.encode()).hexdigest()[:10]
            out.append({
                "scenario_id": f"synthetic_{sid}",
                "origin": "synthetic_expansion_from_book_seed",
                "seed_id": seed.get("seed_id"),
                "title": f'{seed.get("title", "Scenario")} — {context[0]} variation {i+1}',
                "situation_summary": seed.get("situation_summary", "") + " " + context[1] + ".",
                "presenting_problem": seed.get("presenting_problem", ""),
                "typical_thoughts_or_feelings": seed.get("typical_thoughts_or_feelings", []),
                "observable_behaviors": seed.get("observable_behaviors", []),
                "initial_anxiety_estimate": seed.get("initial_anxiety_estimate", 5),
                "anxiety_trajectory": trajectory,
                "modality_relevance": seed.get("modality_relevance", []),
                "language": seed.get("language", "und"),
                "source_refs": seed.get("source_refs", []),
                "is_synthetic": True,
                "review_status": "needs_human_review"
            })
    return out

def build_outputs(output_dir: Path, expanded_per_seed=3):
    raw_strategies = load_json(output_dir / "strategies_raw.json")
    seeds = load_json(output_dir / "scenario_seeds.json")
    strategies = deduplicate_strategies(raw_strategies)
    scenarios = build_scenarios(seeds, expanded_per_seed)
    (output_dir / "strategies.json").write_text(json.dumps(strategies, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scenarios.json").write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    rubric = {
      "rubric_id": "project_anxiety_observation_v1",
      "scale": {"minimum": 1, "maximum": 10, "interpretation": "observed/self-reported distress estimate; not a diagnosis or validated clinical instrument"},
      "levels": [
        {"range": [1,2], "label": "low", "description": "Little current distress; able to reflect and engage."},
        {"range": [3,4], "label": "mild", "description": "Noticeable worry or tension with generally preserved functioning."},
        {"range": [5,6], "label": "moderate", "description": "Distress interferes with concentration or decisions; support and clarification may help."},
        {"range": [7,8], "label": "high", "description": "Strong distress, narrowed attention, or marked avoidance; prioritize pacing, validation, and safety assessment where indicated."},
        {"range": [9,10], "label": "very_high", "description": "Overwhelming distress or inability to function in the moment; assess immediate safety and urgent support needs rather than assuming routine counselling is sufficient."}
      ],
      "scoring_guidance": [
        "Do not infer a numeric anxiety score from writing style alone.",
        "Prefer an explicit client self-rating when available.",
        "Record the evidence and uncertainty behind each score.",
        "A score is not a diagnosis, risk assessment, or substitute for a qualified clinician.",
        "Do not force scores to decline; realistic sessions may remain stable or fluctuate."
      ],
      "source": "Project-defined operational rubric; not represented as a scale authored or validated by Beck or Rogers."
    }
    (output_dir / "anxiety_rubric.json").write_text(json.dumps(rubric, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {len(strategies)} deduplicated strategies and {len(scenarios)} synthetic scenarios.")
