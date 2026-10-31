"""Source-bound communication for three independently configured local models.

This adapter does not load models in this process or grant layout authority.
It uses the existing offline provider's analyze/plan/critique protocols; every
accepted receipt binds the exact request, source image, menu and image inputs.
The graph owns evidence publication and deterministic task verification.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path

from engine.visual_analysis import validate_analysis
from engine.visual_pipeline import (
    _io,
    _source_summary,
    _validated_result,
    make_request,
    run_provider,
)
from engine.visual_plan import source_digest
from experiments.visual_planner.provider import MAX_RESPONSE_BYTES, prepare_request

ROLES = ("analyzer", "organizer", "verifier")
MAX_FEEDBACK_BYTES = 2048
MAX_IDENTITY_FILE_BYTES = 4 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
_PROVIDER_CODE = (
    "experiments/agentic_graph/providers.py",
    "experiments/visual_planner/provider.py",
    "experiments/visual_planner/worker.py",
    "experiments/visual_planner/native.py",
    "engine/visual_pipeline.py",
    "engine/visual_analysis.py",
    "engine/visual_plan.py",
    "engine/visual_selection.py",
)


def _file_identity(path):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_IDENTITY_FILE_BYTES:
        raise ValueError("Model-role configuration/code file is missing or oversized")
    before = path.stat()
    with path.open("rb") as stream:
        data = stream.read(MAX_IDENTITY_FILE_BYTES + 1)
    after = path.stat()
    if len(data) > MAX_IDENTITY_FILE_BYTES or (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Model-role configuration/code changed while recording its identity")
    import hashlib

    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


class LocalModelRoles:
    """Call three role commands and return bounded JSON communication evidence.

    Config has ``analyzer``, ``organizer`` and ``verifier`` command arrays (or
    objects with a ``command`` array), and optional ``timeout_seconds`` (default
    180, maximum 600). A test runner can replace ``run_provider`` without loading
    a model. Returned ``executed`` means a source-bound execution receipt passed;
    it never establishes task quality or fine-tuned adapter qualification.
    """

    def __init__(self, config, *, runner=None):
        if not isinstance(config, dict):
            raise ValueError("Model-role configuration must be an object")
        timeout = config.get("timeout_seconds", 180)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("Role timeout must be a finite number in (0, 600]")
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("Role timeout must be a finite number in (0, 600]")
        self._configuration_sha256 = _io._digest(config)
        self.timeout_seconds = timeout
        self.commands = {}
        for role in ROLES:
            value = config.get(role, config.get(f"{role}_command"))
            if isinstance(value, dict):
                value = value.get("command")
            self.commands[role] = _io.model_command(value)
        self.execution_scope = (
            "supervised_offline_runtime" if runner is None else "injected_provider_test"
        )
        self.runner = runner or run_provider

    def configuration_identity(self):
        """Snapshot role configuration and provider code without reading weights.

        Call before a run and at role boundaries. Changes to a command, its
        ``--config`` file (also supporting ``--config=path``), timeout or provider
        implementation change this identity. Config contents, raw argument
        values and environment variables are never included in the result.
        """
        roles = {}
        for role in ROLES:
            command = self.commands[role]
            paths = []
            for index, argument in enumerate(command):
                if argument == "--config":
                    if index + 1 >= len(command) or command[index + 1].startswith("--"):
                        raise ValueError("Model-role --config requires a local file path")
                    paths.append(command[index + 1])
                elif argument.startswith("--config="):
                    paths.append(argument.partition("=")[2])
            if any(not path or "://" in path for path in paths):
                raise ValueError("Model-role --config requires a local file path")
            roles[role] = {
                "command_sha256": _io._digest(command),
                "configuration_files": [_file_identity(path) for path in paths],
            }
        identity = {
            "schema_version": 1,
            "execution_scope": self.execution_scope,
            "configuration_sha256": self._configuration_sha256,
            "timeout_seconds": self.timeout_seconds,
            "roles": roles,
            "provider_code_sha256": {
                name: _file_identity(ROOT / name)["sha256"] for name in _PROVIDER_CODE
            },
        }
        identity["digest"] = _io._digest(identity)
        return identity

    def analyze(self, source, source_render, *, timeout_seconds=None):
        """Observe the original only: no target, plan, candidate or repair input."""

        def request():
            return {
                "schema_version": 1,
                "mode": "analyze",
                "source_digest": source_digest(source),
                "source_render": deepcopy(source_render),
                "source_summary": _source_summary(source),
            }

        return self._invoke("analyzer", source, request, timeout_seconds)

    def organize(
        self, source, source_render, menu, analysis=None, feedback=None, *, timeout_seconds=None
    ):
        """Select one trusted plan; repair feedback is bounded advisory data."""

        def request():
            value = make_request(source, source_render, menu, mode="plan", source_analysis=analysis)
            if feedback is not None:
                if not isinstance(feedback, dict):
                    raise ValueError("Repair feedback must be a bounded JSON object")
                if len(_io._canonical(feedback).encode("utf-8")) > MAX_FEEDBACK_BYTES:
                    raise ValueError("Repair feedback exceeds its 2048-byte budget")
                if feedback.get("source_digest", value["source_digest"]) != value["source_digest"]:
                    raise ValueError("Repair feedback belongs to a different source")
                value["source_summary"]["repair_feedback"] = deepcopy(feedback)
                value.pop("request_digest", None)
                value["request_digest"] = _io._digest(value)
            return value

        return self._invoke("organizer", source, request, timeout_seconds)

    def verify(
        self, source, source_render, critic_menu, renders, analysis=None, *, timeout_seconds=None
    ):
        """Compare exactly two admitted rendered choices; this remains advisory."""

        def request():
            return make_request(
                source,
                source_render,
                critic_menu,
                mode="critique",
                renders=renders,
                source_analysis=analysis,
            )

        return self._invoke("verifier", source, request, timeout_seconds)

    def _timeout(self, remaining):
        if remaining is None:
            return self.timeout_seconds
        if (
            isinstance(remaining, bool)
            or not isinstance(remaining, (int, float))
            or not math.isfinite(remaining)
            or remaining <= 0
        ):
            raise ValueError("The graph has no positive finite model-call time budget")
        return min(self.timeout_seconds, remaining)

    def _invoke(self, role, source, build_request, remaining):
        result = {
            "role": role,
            "executed": False,
            "valid": False,
            "abstained": False,
            "request": None,
            "trace": {
                "invoked": False,
                "execution_attested": False,
                "receipt_valid": False,
                "command_digest": _io._digest(self.commands[role]),
            },
        }
        try:
            timeout = self._timeout(remaining)
            deadline = time.monotonic() + timeout
            request = build_request()
            result["request"] = deepcopy(request)
            original_source_digest = source_digest(source)
            if request["source_digest"] != original_source_digest:
                raise ValueError("Model menu belongs to a different original source")
            prepared = prepare_request(request)
            transmitted = deepcopy(request)
            timeout = self._timeout(deadline - time.monotonic())
            result["trace"].update(invoked=True, timeout_seconds=timeout)
            response = self.runner(self.commands[role], transmitted, timeout)
            if transmitted != request or source_digest(source) != original_source_digest:
                raise ValueError("Model communication mutated its bound source/request")
            if len(_io._canonical(response).encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise ValueError("Model response exceeds its byte budget")
            result["response"] = deepcopy(response)
            receipt = response.get("receipt") if isinstance(response, dict) else None
            if isinstance(receipt, dict):
                result["trace"]["receipt"] = deepcopy(receipt)
                result["trace"]["execution_attested"] = receipt.get("executed") is True
            self._receipt(receipt, request, prepared)
            # Actual file bytes must remain pinned through inference as well.
            if prepare_request(request)["images"] != prepared["images"]:
                raise ValueError("Model image inputs changed during execution")
            result["executed"] = True
            result["trace"]["receipt_valid"] = True
            if role == "analyzer":
                if not isinstance(response, dict) or set(response) != {"analysis", "receipt"}:
                    raise ValueError("Source analyzer must return only analysis and receipt")
                analysis = validate_analysis(
                    response["analysis"],
                    source_digest=request["source_digest"],
                    source_render_sha256=request["source_render"]["sha256"],
                    known_element_ids=prepared["known_element_ids"],
                )
                result.update(analysis=analysis, abstained=analysis["abstain"])
            else:
                result["decision"] = _validated_result(response, request["plan_menu"])
            if time.monotonic() > deadline:
                raise ValueError("Model role exhausted its graph time budget")
            result["valid"] = True
        except (
            ValueError,
            OSError,
            UnicodeError,
            RecursionError,
            RuntimeError,
            MemoryError,
            TypeError,
            KeyError,
        ) as exc:
            result["trace"]["fallback_reason"] = str(exc)[:600]
        return result

    @staticmethod
    def _receipt(receipt, request, prepared):
        if not isinstance(receipt, dict) or receipt.get("executed") is not True:
            raise ValueError("Provider did not attest actual model execution")
        expected = {
            "request_digest": _io._digest(request),
            "source_digest": request["source_digest"],
            "source_render_sha256": prepared["images"][0]["sha256"],
            "menu_digest": prepared["menu"]["menu_digest"],
            "target_canvas": prepared["target"],
            "images": prepared["images"],
            "source_analysis_digest": (
                _io._digest(request["source_analysis"]) if "source_analysis" in request else None
            ),
            "input_image_count": len(prepared["images"]),
        }
        if (
            not expected.keys() <= receipt.keys()
            or type(receipt.get("input_image_count")) is not int
            or any(receipt.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(
                "Provider receipt does not bind the exact role request and image inputs"
            )
