"""Source reasoning -> organization -> verification -> bounded repair.

Model messages choose existing plans, never geometry or acceptance thresholds.
Every run has a new artifact directory and durable, hash-linked messages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from engine.resources import using_resources
from engine.visual_pipeline import _engine_api, _implementation_hashes, _io, _winner


def digest(value):
    return hashlib.sha256(_io._canonical(value).encode()).hexdigest()


def write_json(path, value):
    """Publish a complete state file; partial writes never replace the last state."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Graph artifacts cannot replace symlinks")
    temporary = path.with_name(path.name + "." + uuid4().hex + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Messages:
    def __init__(self, directory, source, target):
        self.path = directory / "messages.jsonl"
        self.source, self.target = digest(source), digest(target)
        self.sequence, self.previous = 0, None

    def send(self, sender, recipient, kind, payload, round_number=0):
        self.sequence += 1
        message = {
            "schema_version": 1,
            "sequence": self.sequence,
            "previous_message_sha256": self.previous,
            "source_sha256": self.source,
            "target_sha256": self.target,
            "sender": sender,
            "recipient": recipient,
            "kind": kind,
            "round": round_number,
            "payload": deepcopy(payload),
        }
        message["message_sha256"] = digest(message)
        encoded = _io._canonical(message)
        if len(encoded.encode()) > 512 * 1024:
            raise ValueError("Graph communication exceeded its message budget")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous = message["message_sha256"]
        return message


@dataclass(frozen=True)
class TaskCriteria:
    minimum_font_px: float = 12.0
    minimum_metrics: dict = field(
        default_factory=lambda: {"collision_free": 0.99, "hierarchy": 1.0, "containment": 1.0}
    )

    def __post_init__(self):
        if (
            isinstance(self.minimum_font_px, bool)
            or not isinstance(self.minimum_font_px, (int, float))
            or not math.isfinite(self.minimum_font_px)
            or not 1 <= self.minimum_font_px <= 512
        ):
            raise ValueError("minimum_font_px must be finite and between 1 and 512")
        if not isinstance(self.minimum_metrics, dict) or len(self.minimum_metrics) > 20:
            raise ValueError("Task metrics must be a bounded name-to-threshold mapping")
        for name, value in self.minimum_metrics.items():
            if (
                not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Task metric thresholds must be finite values in [0,1]")


@dataclass(frozen=True)
class GraphLimits:
    max_rounds: int = 4
    max_seconds: float = 600.0
    max_stalled_rounds: int = 2

    def __post_init__(self):
        if type(self.max_rounds) is not int or not 1 <= self.max_rounds <= 8:
            raise ValueError("Choose between one and eight repair rounds")
        if (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not math.isfinite(self.max_seconds)
            or not 1 <= self.max_seconds <= 3600
        ):
            raise ValueError("Graph time budget must be between 1 and 3600 seconds")
        if type(self.max_stalled_rounds) is not int or not 1 <= self.max_stalled_rounds <= 4:
            raise ValueError("Choose between one and four stalled rounds")


def task_verdict(design, verification, criteria):
    failures = [deepcopy(row) for row in verification.get("failures", [])]
    hard = verification.get("hard_pass") is True
    if not hard and not failures:
        failures.append({"code": "hard_evidence_unverified", "element_ids": []})
    small = [
        row["id"]
        for row in design.get("elements", [])
        if row.get("type") == "text"
        and isinstance(row.get("font_size"), (int, float))
        and row["font_size"] < criteria.minimum_font_px
    ]
    if small:
        failures.append(
            {
                "code": "minimum_font",
                "element_ids": small,
                "required": criteria.minimum_font_px,
                "observed": min(
                    row["font_size"] for row in design["elements"] if row["id"] in small
                ),
            }
        )
    metrics = verification.get("diagnostics", {}).get("metrics", {})
    for name, minimum in criteria.minimum_metrics.items():
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
        ):
            failures.append(
                {
                    "code": "metric_" + name,
                    "element_ids": [],
                    "observed": value
                    if isinstance(value, (int, float)) and math.isfinite(value)
                    else None,
                    "required": minimum,
                }
            )
    return {
        "passed": hard and not failures,
        "hard_pass": hard,
        "failures": failures,
        "scope": "explicit fundamental-task checks and local visual proxies",
    }


def _rank(verdict, verification):
    total = verification.get("diagnostics", {}).get("total", 0)
    total = total if isinstance(total, (int, float)) and math.isfinite(total) else 0
    return (verdict["hard_pass"], verdict["passed"], -len(verdict["failures"]), total)


def _feedback(verdict, candidate_id, tried):
    return {
        "candidate_sha256": candidate_id.removeprefix("candidate-"),
        "failed_checks": [
            {
                "code": row["code"],
                "element_ids": row.get("element_ids", [])[:8],
                "observed": row.get("observed"),
                "required": row.get("required"),
                "severity": "task" if row["code"].startswith(("minimum_", "metric_")) else "hard",
            }
            for row in verdict["failures"][:6]
        ],
        "tried_plan_ids": list(tried)[-8:],
        "instruction": "Repair these checks with another allowed plan; preserve source identity.",
    }


def _code_identity():
    from engine.fit_policy import fit_policy_identity

    files = _implementation_hashes()
    files["model_identity:fit_policy"] = digest(fit_policy_identity())
    root = Path(__file__).resolve().parents[2]
    files.update(
        {
            path.relative_to(root).as_posix(): _io._sha(path)
            for path in Path(__file__).parent.glob("*.py")
        }
    )
    return files


def run_graph(
    source,
    target,
    out,
    *,
    assets_dir,
    fonts_dir,
    roles=None,
    criteria=None,
    limits=None,
    source_id=None,
    api=None,
    verifier=None,
    implementation_hashes=None,
):
    """Run one immutable experiment; no models are invoked when roles is None.

    Dependency injection exercises the same communication and acceptance seam in
    tests. Such tests establish loop behavior, not learned visual competence.
    """
    from engine.visual_pipeline import _validate_request
    from experiments.agentic_graph.providers import LocalModelRoles
    from experiments.agentic_graph.verification import verify_design
    from tools.visual_resize_harness import reject_reserved_source

    _validate_request({"source": source, "target_canvas": target})
    reject_reserved_source(source, source_id)
    criteria, limits = criteria or TaskCriteria(), limits or GraphLimits()
    api, verifier = api or _engine_api(), verifier or verify_design
    out = Path(out).resolve()
    for resource in (assets_dir, fonts_dir):
        if out.is_relative_to(Path(resource).resolve()):
            raise ValueError("Graph evidence must remain outside source resource directories")
    out.mkdir(parents=True, exist_ok=False)
    journal = Messages(out, source, target)
    original = deepcopy(source)
    resources = _io.resource_hashes(source, assets_dir, fonts_dir)
    hashes = implementation_hashes or _code_identity
    implementation = hashes()
    role_identity = roles.configuration_identity() if isinstance(roles, LocalModelRoles) else None
    evidence_scope = (
        "live_model_receipts"
        if role_identity is not None
        and role_identity.get("execution_scope") == "supervised_offline_runtime"
        else "explicit_baseline_only"
        if roles is None
        else "injected_protocol_test"
    )
    started, rounds, counts = (
        time.monotonic(),
        [],
        {
            role: {"calls": 0, "executed": 0, "valid": 0}
            for role in ("analyzer", "organizer", "verifier")
        },
    )
    seen_plans, seen_designs, stalled = [], set(), 0
    best, best_verification, best_verdict = None, None, None
    stop_reason, analysis = "baseline_only", None

    def unchanged():
        if source != original or _io.resource_hashes(source, assets_dir, fonts_dir) != resources:
            raise RuntimeError("Source or model-input resources changed during the graph")
        if hashes() != implementation:
            raise RuntimeError("Implementation changed during the graph; keep this run unqualified")
        if role_identity is not None and roles.configuration_identity() != role_identity:
            raise RuntimeError("Model role configuration changed during the graph")

    def remaining():
        return limits.max_seconds - (time.monotonic() - started)

    def checkpoint():
        write_json(
            out / "state.json",
            {
                "schema_version": 1,
                "source_sha256": digest(source),
                "target_canvas": target,
                "rounds_completed": len(rounds),
                "best_design": best,
                "verification": best_verification,
                "task_verdict": best_verdict,
                "role_execution": counts,
                "last_message_sha256": journal.previous,
                "status": "running",
                "resume_policy": "preserved evidence; start a new run to retry",
            },
        )

    def call_role(name, function, directory, **kwargs):
        if remaining() <= 0:
            raise TimeoutError("Graph wall-clock budget exhausted at a role boundary")
        result = function(**kwargs, timeout_seconds=min(600, remaining()))
        if not isinstance(result, dict):
            raise ValueError("A model role returned no structured execution result")
        if result.get("valid") and result.get("executed") is not True:
            result = {
                **result,
                "valid": False,
                "trace": {
                    **result.get("trace", {}),
                    "fallback_reason": "A valid model role must attest actual execution",
                },
            }
        counts[name]["calls"] += 1
        counts[name]["executed"] += result.get("executed") is True
        counts[name]["valid"] += result.get("valid") is True
        write_json(directory / f"{name}.json", result)
        journal.send(
            name,
            "coordinator",
            "role_result",
            {
                "valid": result.get("valid") is True,
                "executed": result.get("executed") is True,
                "decision": result.get("decision"),
                "analysis": result.get("analysis"),
                "fallback_reason": result.get("trace", {}).get("fallback_reason"),
            },
            len(rounds) + 1,
        )
        unchanged()
        return result

    def assess(row, directory, label, *, promote=True):
        nonlocal best, best_verification, best_verdict
        verification = verifier(
            source, row["design"], target, assets_dir=assets_dir, fonts_dir=fonts_dir
        )
        verdict = task_verdict(row["design"], verification, criteria)
        write_json(directory / f"{label}-verification.json", verification)
        write_json(directory / f"{label}-task.json", verdict)
        if promote and (
            best is None or _rank(verdict, verification) > _rank(best_verdict, best_verification)
        ):
            best, best_verification, best_verdict = deepcopy(row["design"]), verification, verdict
        unchanged()
        return verification, verdict

    write_json(out / "source.json", source)
    write_json(
        out / "run-config.json",
        {
            "criteria": asdict(criteria),
            "limits": asdict(limits),
            "resources": resources,
            "implementation": implementation,
            "source_id": source_id,
            "model_mode": "three_roles" if roles is not None else "explicit_baseline_only",
            "role_configuration_identity": role_identity,
            "evidence_scope": evidence_scope,
        },
    )
    try:
        with using_resources(assets_dir, fonts_dir):
            source_render = _io._render(source, out / "source.png")
            source_render["path"] = str(out / "source.png")
            baseline = api["snapshot"](source, target, strict_bounds=True)
            write_json(out / "baseline-snapshot.json", baseline)
            baseline_row = _winner(baseline)
            assess(baseline_row, out, "baseline")
            journal.send("deterministic_verifier", "coordinator", "baseline_checks", best_verdict)
            checkpoint()
            if roles is not None and remaining() <= 0:
                stop_reason = "time_budget"
            if roles is not None and remaining() > 0:
                perception = call_role(
                    "analyzer", roles.analyze, out, source=source, source_render=source_render
                )
                if perception.get("valid") and not perception.get("abstained"):
                    analysis = perception["analysis"]
                    stop_reason = "max_rounds"
                    feedback = _feedback(best_verdict, digest(best), seen_plans)
                    for index in range(1, limits.max_rounds + 1):
                        if remaining() <= 0:
                            stop_reason = "time_budget"
                            break
                        directory = out / f"round-{index:03d}"
                        directory.mkdir()
                        menu = api["menu"](source, target)
                        write_json(directory / "plan-menu.json", menu)
                        if remaining() <= 0:
                            stop_reason = "time_budget"
                            break
                        if menu.get("status") != "ready" or len(menu.get("plans", [])) < 2:
                            stop_reason = "no_organizer_choice"
                            break
                        journal.send("coordinator", "organizer", "repair_feedback", feedback, index)
                        organized = call_role(
                            "organizer",
                            roles.organize,
                            directory,
                            source=source,
                            source_render=source_render,
                            menu=menu,
                            analysis=analysis,
                            feedback=feedback,
                        )
                        if not organized.get("valid"):
                            stop_reason = "invalid_organizer"
                            break
                        decision = organized["decision"]
                        plan_id = decision["plan_id"]
                        if plan_id in seen_plans:
                            stop_reason = "repeated_plan"
                            break
                        seen_plans.append(plan_id)
                        snapshot = api["snapshot"](
                            source, target, visual_plan=decision, strict_bounds=True
                        )
                        write_json(directory / "candidate-snapshot.json", snapshot)
                        if remaining() <= 0:
                            stop_reason = "time_budget"
                            break
                        selected = _winner(snapshot)
                        critic_menu = api["critic_menu"](snapshot)
                        write_json(directory / "verifier-menu.json", critic_menu)
                        old_rank = _rank(best_verdict, best_verification)
                        if len(critic_menu.get("plans", [])) >= 2:
                            renders = {}
                            for choice in critic_menu["plans"]:
                                row = next(
                                    row
                                    for row in snapshot["candidates"]
                                    if row["candidate_id"] == choice["plan_id"]
                                )
                                picture = directory / f"{choice['plan_id']}.png"
                                rendered = _io._render(row["design"], picture)
                                rendered["path"] = str(picture)
                                renders[choice["plan_id"]] = rendered
                                assess(row, directory, choice["plan_id"], promote=False)
                            if remaining() <= 0:
                                stop_reason = "time_budget"
                                break
                            critique = call_role(
                                "verifier",
                                roles.verify,
                                directory,
                                source=source,
                                source_render=source_render,
                                critic_menu=critic_menu,
                                renders=renders,
                                analysis=analysis,
                            )
                            if not critique.get("valid"):
                                stop_reason = "invalid_verifier"
                                break
                            applied = api["select"](
                                source,
                                target,
                                snapshot,
                                critique["decision"],
                                critic_menu=critic_menu,
                            )
                            selected = {"design": applied["design"], "state": applied["state"]}
                        verification, verdict = assess(selected, directory, "selected")
                        identity = digest(selected["design"])
                        round_result = {
                            "round": index,
                            "plan_id": plan_id,
                            "candidate_sha256": identity,
                            "task_verdict": verdict,
                            "verification_status": verification.get("status"),
                        }
                        rounds.append(round_result)
                        journal.send(
                            "deterministic_verifier",
                            "organizer",
                            "task_feedback",
                            _feedback(verdict, identity, seen_plans),
                            index,
                        )
                        checkpoint()
                        if best_verdict["passed"]:
                            stop_reason = "fundamental_tasks_passed"
                            break
                        if identity in seen_designs:
                            stop_reason = "repeated_candidate"
                            break
                        seen_designs.add(identity)
                        stalled = (
                            stalled + 1 if _rank(best_verdict, best_verification) <= old_rank else 0
                        )
                        if stalled >= limits.max_stalled_rounds:
                            stop_reason = "no_progress"
                            break
                        feedback = _feedback(best_verdict, digest(best), seen_plans)
                else:
                    stop_reason = "analyzer_abstained_or_invalid"
            unchanged()
            if best_verdict["hard_pass"]:
                write_json(out / "final-design.json", best)
                _io._render(best, out / "final.png")
            unchanged()
            passed = best_verdict["passed"]
            all_roles = all(counts[role]["valid"] and counts[role]["executed"] for role in counts)
            report = {
                "schema_version": 1,
                "status": "passed"
                if passed and all_roles
                else "baseline_passed"
                if passed
                else "failed",
                "task_passed": passed,
                "stop_reason": stop_reason,
                "design": best,
                "verification": best_verification,
                "task_verdict": best_verdict,
                "rounds": rounds,
                "role_execution": counts,
                "all_three_role_protocols_completed": all_roles,
                "all_three_roles_executed": all_roles and evidence_scope == "live_model_receipts",
                "evidence_scope": evidence_scope,
                "model_layout_selected": best != baseline_row["design"],
                "trained_model_quality": "not_qualified_by_protocol_or_proxy_checks",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "last_message_sha256": journal.previous,
                "source_unchanged": True,
                "resources_unchanged": True,
                "implementation_unchanged": True,
            }
            write_json(out / "report.json", report)
            write_json(out / "state.json", report)
            return report
    except Exception as exc:
        failure = {
            "status": "aborted",
            "error": str(exc)[:1000],
            "rounds": rounds,
            "role_execution": counts,
            "best_design": best,
            "verification": best_verification,
            "task_verdict": best_verdict,
        }
        write_json(out / "failure.json", failure)
        write_json(out / "state.json", failure)
        raise
