"""Validation-only checkpoint choice with the unchanged student retained as step zero."""
import math


def choose_checkpoint(metrics):
    if set(metrics) != {0, 50, 100, 150}:
        raise ValueError("Require all four planned validation checkpoints")
    for result in metrics.values():
        if (result["n"] != 1390 or not math.isfinite(result["mean_kl"]) or result["mean_kl"] < -1e-6
                or not math.isfinite(result["teacher_agreement"]) or not 0 <= result["teacher_agreement"] <= 1):
            raise ValueError("Checkpoint validation must be complete and finite")
    floor = metrics[0]["teacher_agreement"] - .01
    eligible = sorted(step for step, result in metrics.items() if result["teacher_agreement"] >= floor)
    selected = min(eligible, key=lambda step: (metrics[step]["mean_kl"], step))
    return {"selected_step": selected, "eligible_steps": eligible, "teacher_agreement_floor": floor,
            "selection": "Minimum original-validation mean KL among checkpoints losing at most 0.01 teacher agreement versus step zero; exact ties prefer the earlier step. All public and authored confirmation data are excluded."}
