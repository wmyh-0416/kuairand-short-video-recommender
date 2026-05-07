from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.experiments.metrics import GroundTruth, evaluate_group
from src.experiments.policy import PolicyContext, RecommendationPolicy, build_policy
from src.utils.paths import artifacts_dir, ensure_dir, processed_dir


def stable_bucket(user_id: int, seed: int) -> float:
    payload = f"{seed}:{int(user_id)}".encode("utf-8")
    digest = hashlib.md5(payload).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def assign_groups(
    user_ids: list[int],
    traffic_split: Mapping[str, float],
    seed: int,
) -> dict[int, str]:
    normalized = {str(name): float(value) for name, value in traffic_split.items()}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("traffic_split must contain positive weights")
    normalized = {name: value / total for name, value in normalized.items()}

    thresholds: list[tuple[str, float]] = []
    cumulative = 0.0
    for name, weight in normalized.items():
        cumulative += weight
        thresholds.append((name, cumulative))

    assignments: dict[int, str] = {}
    for user_id in user_ids:
        bucket = stable_bucket(int(user_id), seed=seed)
        group_name = thresholds[-1][0]
        for name, threshold in thresholds:
            if bucket < threshold:
                group_name = name
                break
        assignments[int(user_id)] = group_name
    return assignments


def _build_ground_truth(split_df: pd.DataFrame, item_tag_map: Mapping[int, str]) -> GroundTruth:
    if "is_positive" in split_df.columns:
        positive_mask = split_df["is_positive"].fillna(0).astype(int) > 0
    else:
        positive_mask = pd.Series(False, index=split_df.index)
        for col in ["long_watch", "finish", "like", "is_click"]:
            if col in split_df.columns:
                positive_mask = positive_mask | (split_df[col].fillna(0).astype(int) > 0)

    if "long_watch" in split_df.columns:
        long_view_mask = split_df["long_watch"].fillna(0).astype(int) > 0
    elif "long_view" in split_df.columns:
        long_view_mask = split_df["long_view"].fillna(0).astype(int) > 0
    elif {"play_time_ms", "duration_ms"} <= set(split_df.columns):
        ratio = split_df["play_time_ms"] / split_df["duration_ms"].replace(0, np.nan)
        long_view_mask = ratio.fillna(0.0) >= 0.7
    else:
        long_view_mask = pd.Series(False, index=split_df.index)

    if "like" in split_df.columns:
        like_mask = split_df["like"].fillna(0).astype(int) > 0
    elif "is_like" in split_df.columns:
        like_mask = split_df["is_like"].fillna(0).astype(int) > 0
    else:
        like_mask = pd.Series(False, index=split_df.index)

    if "is_click" in split_df.columns:
        click_mask = split_df["is_click"].fillna(0).astype(int) > 0
    else:
        click_mask = pd.Series(False, index=split_df.index)

    def _to_user_item_map(mask: pd.Series) -> dict[int, set[int]]:
        filtered = split_df.loc[mask, ["user_id", "video_id"]]
        user_item_map: dict[int, set[int]] = {}
        for user_id, group in filtered.groupby("user_id", sort=False):
            user_item_map[int(user_id)] = {int(video_id) for video_id in group["video_id"].tolist()}
        return user_item_map

    impression_counts = {
        int(user_id): int(count)
        for user_id, count in split_df.groupby("user_id", sort=False).size().items()
    }
    catalog_size = int(len(item_tag_map)) if item_tag_map else int(split_df["video_id"].nunique())
    return GroundTruth(
        positive_items=_to_user_item_map(positive_mask),
        long_view_items=_to_user_item_map(long_view_mask),
        like_items=_to_user_item_map(like_mask),
        click_items=_to_user_item_map(click_mask),
        impression_counts=impression_counts,
        catalog_size=catalog_size,
        item_tag_map={int(k): str(v) for k, v in item_tag_map.items()},
    )


def _relative_lift(control_value: Any, treatment_value: Any) -> float | None:
    if control_value is None or treatment_value is None:
        return None
    try:
        control = float(control_value)
        treatment = float(treatment_value)
    except (TypeError, ValueError):
        return None
    if abs(control) < 1e-12:
        return None
    return float((treatment - control) / control)


def bootstrap_primary_metric(
    control_values: list[float],
    treatment_values: list[float],
    *,
    n_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if len(control_values) < 5 or len(treatment_values) < 5:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "insufficient_sample_size",
            "control_users": int(len(control_values)),
            "treatment_users": int(len(treatment_values)),
        }

    rng = np.random.default_rng(seed)
    control = np.asarray(control_values, dtype=np.float64)
    treatment = np.asarray(treatment_values, dtype=np.float64)

    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    observed_delta = float(treatment_mean - control_mean)
    observed_relative_lift = _relative_lift(control_mean, treatment_mean)

    deltas: list[float] = []
    relative_lifts: list[float] = []
    for _ in range(int(n_samples)):
        control_sample = rng.choice(control, size=control.shape[0], replace=True)
        treatment_sample = rng.choice(treatment, size=treatment.shape[0], replace=True)
        sampled_control = float(control_sample.mean())
        sampled_treatment = float(treatment_sample.mean())
        deltas.append(float(sampled_treatment - sampled_control))
        rel = _relative_lift(sampled_control, sampled_treatment)
        if rel is not None:
            relative_lifts.append(float(rel))

    alpha = 1.0 - float(confidence_level)
    delta_ci = [
        float(np.quantile(deltas, alpha / 2)),
        float(np.quantile(deltas, 1.0 - alpha / 2)),
    ]
    result: dict[str, Any] = {
        "enabled": True,
        "skipped": False,
        "n_samples": int(n_samples),
        "confidence_level": float(confidence_level),
        "observed_control": control_mean,
        "observed_treatment": treatment_mean,
        "observed_delta": observed_delta,
        "delta_ci": delta_ci,
    }
    if relative_lifts and observed_relative_lift is not None:
        result["observed_relative_lift"] = float(observed_relative_lift)
        result["bootstrap_ci_95"] = [
            float(np.quantile(relative_lifts, alpha / 2)),
            float(np.quantile(relative_lifts, 1.0 - alpha / 2)),
        ]
    else:
        result["observed_relative_lift"] = observed_relative_lift
        result["bootstrap_ci_95"] = None
    return result


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    return value


def _policy_summary(policy: RecommendationPolicy) -> dict[str, Any]:
    return policy.metadata()


def _build_summary_markdown(report: Mapping[str, Any]) -> str:
    experiment = report["experiment"]
    metrics_cfg = report["metrics"]
    assignment = report["group_assignment_summary"]
    control = report["control"]
    treatment = report["treatment"]
    policy_artifacts = report["policy_artifacts"]
    lift = report["relative_lift"]
    bootstrap = report["bootstrap"]

    topks = experiment["top_k"]
    core_metrics = [metrics_cfg["primary"]] + [metric for metric in metrics_cfg.get("secondary", []) if metric not in {metrics_cfg["primary"]}]

    lines = [
        "# Offline A/B Test Simulation",
        "",
        "This report is an **offline log-replay simulation**, not a real online A/B test.",
        "",
        "## Experiment Setup",
        "",
        f"- split: `{experiment['split']}`",
        f"- random_seed: `{experiment['random_seed']}`",
        f"- group_by: `{experiment['group_by']}`",
        f"- top_k evaluated: `{topks}`",
        f"- control: `{control['policy']['name']}` (`{control['policy']['type']}`)",
        f"- treatment: `{treatment['policy']['name']}` (`{treatment['policy']['type']}`)",
        "",
        "## Traffic Split",
        "",
        f"- control users: `{assignment['control']['user_count']}` impressions: `{assignment['control']['impression_count']}`",
        f"- treatment users: `{assignment['treatment']['user_count']}` impressions: `{assignment['treatment']['impression_count']}`",
        "",
        "## Core Metrics",
        "",
        "| metric | control | treatment | relative_lift |",
        "|---|---:|---:|---:|",
    ]
    for metric_name in core_metrics:
        control_value = control["metrics"].get(metric_name)
        treatment_value = treatment["metrics"].get(metric_name)
        lift_value = lift.get(metric_name)
        control_text = "n/a" if control_value is None else f"{float(control_value):.6f}"
        treatment_text = "n/a" if treatment_value is None else f"{float(treatment_value):.6f}"
        lift_text = "n/a" if lift_value is None else f"{float(lift_value):.4%}"
        lines.append(f"| {metric_name} | {control_text} | {treatment_text} | {lift_text} |")

    lines.extend(
        [
            "",
            "## Bootstrap",
            "",
            f"- primary metric: `{metrics_cfg['primary']}`",
        ]
    )
    if bootstrap.get("skipped"):
        lines.append(f"- bootstrap skipped: `{bootstrap.get('reason', 'unknown')}`")
    else:
        lines.append(f"- observed control: `{bootstrap.get('observed_control', 0.0):.6f}`")
        lines.append(f"- observed treatment: `{bootstrap.get('observed_treatment', 0.0):.6f}`")
        lines.append(f"- observed relative lift: `{bootstrap.get('observed_relative_lift', 0.0):.4%}`")
        ci = bootstrap.get("bootstrap_ci_95")
        if ci is not None:
            lines.append(f"- bootstrap_ci_95 (relative lift): `[{ci[0]:.4f}, {ci[1]:.4f}]`")

    lines.extend(
        [
            "",
            "## Artifacts Used",
            "",
            f"- control artifacts: `{[entry['path'] for entry in policy_artifacts['control']['artifacts'] if entry['available']]}`",
            f"- treatment artifacts: `{[entry['path'] for entry in policy_artifacts['treatment']['artifacts'] if entry['available']]}`",
            "",
            "## Limitations",
            "",
            "- This is offline log-replay evaluation on the held-out split, not a real online randomized experiment.",
            "- It cannot correct exposure bias or estimate true causal CTR / watch-time lift.",
            "- Policies may fallback to earlier-stage artifacts when later-stage artifacts are missing, and that fallback is recorded in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_offline_ab_test(
    cfg: Mapping[str, Any],
    component_cfgs: Mapping[str, Mapping[str, Any]],
    logger: Any | None = None,
) -> dict[str, Any]:
    experiment_cfg = cfg["experiment"]
    metrics_cfg = cfg["metrics"]
    split = str(experiment_cfg.get("split", "test"))
    topks = sorted({int(k) for k in experiment_cfg.get("top_k", [10, 50])})
    seed = int(experiment_cfg.get("random_seed", 2026))

    recall_cfg = component_cfgs["recall"]
    split_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["splits_dir"] / f"{split}.parquet"
    user_seq_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["user_sequences_file"]
    item_path = processed_dir(recall_cfg) / recall_cfg["recall"]["processed"]["item_features_file"]

    split_df = pd.read_parquet(split_path)
    user_sequences = pd.read_parquet(user_seq_path, columns=["user_id", "watch_seq", "history_len", "last_time_ms"])
    item_features = pd.read_parquet(item_path, columns=["video_id", "author_id", "tag"])

    context = PolicyContext(
        experiment_cfg=cfg,
        component_cfgs=component_cfgs,
        split=split,
        max_top_k=max(topks),
        logger=logger,
        user_sequences=user_sequences,
        item_features=item_features,
    )

    control_policy = build_policy(cfg["policies"]["control"], context=context, logger=logger)
    treatment_policy = build_policy(cfg["policies"]["treatment"], context=context, logger=logger)

    user_ids = sorted(int(user_id) for user_id in split_df[str(experiment_cfg.get("group_by", "user_id"))].unique().tolist())
    assignments = assign_groups(user_ids, experiment_cfg["traffic_split"], seed=seed)

    ground_truth = _build_ground_truth(split_df=split_df, item_tag_map=context.item_tag_map)
    group_users: dict[str, list[int]] = {}
    for user_id in user_ids:
        group_name = assignments[int(user_id)]
        group_users.setdefault(group_name, []).append(int(user_id))

    def _collect_recommendations(policy: RecommendationPolicy, users: list[int]) -> dict[int, list[Any]]:
        results: dict[int, list[Any]] = {}
        for idx, user_id in enumerate(users, start=1):
            results[int(user_id)] = policy.recommend_with_details(int(user_id), max(topks))
            if logger and idx % 5000 == 0:
                logger.info("Policy %s replay progress: %d/%d users", policy.name, idx, len(users))
        return results

    control_users = group_users.get("control", [])
    treatment_users = group_users.get("treatment", [])
    control_recommendations = _collect_recommendations(control_policy, control_users)
    treatment_recommendations = _collect_recommendations(treatment_policy, treatment_users)

    control_eval = evaluate_group(control_recommendations, control_users, ground_truth, topks)
    treatment_eval = evaluate_group(treatment_recommendations, treatment_users, ground_truth, topks)

    metric_names = sorted(set(control_eval["metrics"]) | set(treatment_eval["metrics"]))
    relative_lift = {
        metric_name: _relative_lift(control_eval["metrics"].get(metric_name), treatment_eval["metrics"].get(metric_name))
        for metric_name in metric_names
    }

    primary_metric = str(metrics_cfg.get("primary", f"long_view_rate@{topks[0]}"))
    bootstrap_cfg = cfg.get("statistics", {}).get("bootstrap", {})
    bootstrap_result = {
        "enabled": False,
        "skipped": True,
        "reason": "bootstrap_disabled",
    }
    if bool(bootstrap_cfg.get("enabled", True)):
        bootstrap_result = bootstrap_primary_metric(
            control_values=control_eval["per_user_metrics"].get(primary_metric, []),
            treatment_values=treatment_eval["per_user_metrics"].get(primary_metric, []),
            n_samples=int(bootstrap_cfg.get("n_samples", 500)),
            confidence_level=float(bootstrap_cfg.get("confidence_level", 0.95)),
            seed=seed,
        )

    warnings: list[str] = []
    warnings.extend(control_policy.warnings)
    warnings.extend(treatment_policy.warnings)
    if control_policy.missing_artifacts or treatment_policy.missing_artifacts:
        warnings.append("One or more policy artifacts were missing; see missing_artifacts for explicit fallback details.")

    report: dict[str, Any] = {
        "experiment": {
            "name": str(experiment_cfg.get("name", "offline_ab_test")),
            "split": split,
            "random_seed": seed,
            "group_by": str(experiment_cfg.get("group_by", "user_id")),
            "top_k": topks,
            "traffic_split": {str(k): float(v) for k, v in experiment_cfg["traffic_split"].items()},
        },
        "metrics": {
            "primary": primary_metric,
            "secondary": list(metrics_cfg.get("secondary", [])),
        },
        "policy_artifacts": {
            "control": _policy_summary(control_policy),
            "treatment": _policy_summary(treatment_policy),
        },
        "missing_artifacts": control_policy.missing_artifacts + treatment_policy.missing_artifacts,
        "group_assignment_summary": {
            "control": {
                "user_count": int(len(control_users)),
                "impression_count": int(sum(ground_truth.impression_counts.get(user_id, 0) for user_id in control_users)),
            },
            "treatment": {
                "user_count": int(len(treatment_users)),
                "impression_count": int(sum(ground_truth.impression_counts.get(user_id, 0) for user_id in treatment_users)),
            },
        },
        "control": {
            "policy": {"name": control_policy.name, "type": control_policy.policy_type},
            "metrics": control_eval["metrics"],
        },
        "treatment": {
            "policy": {"name": treatment_policy.name, "type": treatment_policy.policy_type},
            "metrics": treatment_eval["metrics"],
        },
        "relative_lift": relative_lift,
        "bootstrap": bootstrap_result,
        "warnings": warnings,
        "notes": [
            "This is an offline log-replay simulation over the held-out split.",
            "It is useful for resume-ready experimentation narratives, but it is not a real online randomized A/B test.",
        ],
    }

    output_cfg = cfg.get("output", {})
    experiments_path = ensure_dir(artifacts_dir(cfg) / output_cfg.get("experiments_dir", "experiments"))
    report_path = experiments_path / output_cfg.get("report_file", "ab_test_report.json")
    summary_path = experiments_path / output_cfg.get("summary_file", "ab_test_summary.md")

    report_path.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=False), encoding="utf-8")
    summary_path.write_text(_build_summary_markdown(report), encoding="utf-8")

    if logger:
        logger.info("Saved offline A/B report: %s", report_path)
        logger.info("Saved offline A/B summary: %s", summary_path)
    return report
