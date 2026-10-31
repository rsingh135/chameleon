"""Diagnostic graph verifier using an operator-scored saved A/B/N adapter.

Only this verifier is replaced. Existing analyzer/organizer roles, candidate
admission, and native selection remain unchanged. No model process is started.
"""

from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from pathlib import Path

from engine.visual_analysis import validate_analysis

from .adapter_analysis import _identity, _source, digest, source_digest
from .adapter_critic import (
    build_adapter_critic_request,
    complete_adapter_critic,
    verify_render_files,
)

MODEL_KEYS = {"base", "adapter_sha256", "adapter_config_sha256", "code_sha256"}


def create_adapter_critic_roles(
    delegated_roles, *, score_runner, model_identity, driver_identity, mode="off"
):
    """Create an off-by-default graph role; the operator owns model execution.

    ``score_runner(bundle, timeout_seconds=...)`` returns exact prediction bytes,
    the task-specific receipt, and ``prepared_identity`` frozen BEFORE inference.
    Native source and both candidate PNGs are byte-checked before and after it.
    """
    from .providers import LocalModelRoles

    if mode not in {"off", "diagnostic"}:
        raise ValueError("Unqualified critic bridge supports off or diagnostic only")
    if not isinstance(model_identity, dict) or set(model_identity) != MODEL_KEYS:
        raise ValueError("Critic role must pin base/adapter/config/worker code before inference")
    _identity({**model_identity, "recipe_sha256": "0" * 64, "dataset_sha256": "0" * 64})
    if not callable(score_runner) or not isinstance(driver_identity, dict):
        raise ValueError("Critic role needs an explicit operator scoring callback and identity")
    digest(driver_identity)

    class AdapterCriticRoles(LocalModelRoles):
        def __init__(self):
            self._delegated = delegated_roles
            self._model = deepcopy(model_identity)
            self._driver_identity = deepcopy(driver_identity)
            self._mode = mode
            self.execution_scope = (
                "supervised_offline_runtime"
                if getattr(delegated_roles, "execution_scope", None) == "supervised_offline_runtime"
                and driver_identity.get("execution_scope") == "supervised_offline_runtime"
                else "injected_critic_adapter_protocol_test"
            )

        def configuration_identity(self):
            value = {
                "schema_version": 1,
                "execution_scope": self.execution_scope,
                "delegated_analyzer_organizer": self._delegated.configuration_identity(),
                "verifier": {
                    "kind": "saved_adapter_abn_classifier",
                    "mode": self._mode,
                    "model_identity": deepcopy(self._model),
                    "driver_identity": deepcopy(self._driver_identity),
                    "bridge_code_sha256": {
                        name: hashlib.sha256(
                            Path(__file__).with_name(name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "adapter_critic.py",
                            "adapter_critic_roles.py",
                            "adapter_analysis.py",
                        )
                    },
                    "deployment_enabled": False,
                    "qualified": False,
                },
            }
            value["digest"] = digest(value)
            return value

        def analyze(self, *args, **kwargs):
            return self._delegated.analyze(*args, **kwargs)

        def organize(self, *args, **kwargs):
            return self._delegated.organize(*args, **kwargs)

        def verify(
            self,
            source,
            source_render,
            critic_menu,
            renders,
            analysis=None,
            *,
            timeout_seconds=None,
        ):
            result = {
                "role": "verifier",
                "executed": False,
                "valid": False,
                "prediction_valid": False,
                "abstained": True,
                "decision": None,
                "trace": {
                    "invoked": False,
                    "deployment_enabled": False,
                    "qualified": False,
                    "experimental_only": True,
                    "mode": self._mode,
                },
            }
            if self._mode == "off":
                result["trace"]["fallback_reason"] = "adapter_critic_disabled"
                return result
            try:
                seconds = 180 if timeout_seconds is None else timeout_seconds
                if (
                    type(seconds) not in {int, float}
                    or not math.isfinite(seconds)
                    or not 0 < seconds <= 600
                ):
                    raise ValueError("Critic graph timeout must be finite and within (0,600]")
                started = time.monotonic()
                source_pin = source_digest(source)
                input_pin = digest({"menu": critic_menu, "renders": renders})
                bundle = build_adapter_critic_request(source, source_render, critic_menu, renders)
                verify_render_files(bundle)
                analysis_pin = None
                if analysis is not None:
                    validated = validate_analysis(
                        analysis,
                        source_digest=source_pin,
                        source_render_sha256=bundle["source_render_sha256"],
                        known_element_ids=_source(source),
                    )
                    analysis_pin = digest(validated)
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise ValueError("Critic graph time budget expired before scoring")
                transmitted = deepcopy(bundle)
                result["trace"]["invoked"] = True
                scored = score_runner(transmitted, timeout_seconds=remaining)
                if (
                    transmitted != bundle
                    or source_digest(source) != source_pin
                    or digest({"menu": critic_menu, "renders": renders}) != input_pin
                ):
                    raise ValueError(
                        "Critic scoring mutated its source/menu/render/request context"
                    )
                required = {"prediction_bytes", "receipt", "prepared_identity"}
                if (
                    not isinstance(scored, dict)
                    or not required <= scored.keys()
                    or scored.keys() - required - {"base_prediction_bytes"}
                ):
                    raise ValueError(
                        "Critic driver must return scores, receipt and prepared identity"
                    )
                prepared = _identity(scored["prepared_identity"])
                if any(prepared[key] != value for key, value in self._model.items()):
                    raise ValueError(
                        "Critic driver changed the operator-pinned model/code identity"
                    )
                result = complete_adapter_critic(
                    source,
                    source_render,
                    critic_menu,
                    renders,
                    bundle,
                    scored["prediction_bytes"],
                    scored["receipt"],
                    expected_identity=prepared,
                    mode=self._mode,
                    base_prediction_bytes=scored.get("base_prediction_bytes"),
                )
                verify_render_files(bundle)
                result["trace"].update(
                    driver_identity=deepcopy(self._driver_identity),
                    source_analysis_digest=analysis_pin,
                    source_analysis_used_by_classifier=False,
                )
                if time.monotonic() - started > seconds:
                    raise ValueError("Critic adapter exhausted its graph deadline")
            except (ValueError, OSError, KeyError, TypeError, UnicodeError, RecursionError) as exc:
                result.update(valid=False, abstained=True, decision=None)
                result["trace"]["fallback_reason"] = str(exc)[:600]
            return result

    return AdapterCriticRoles()
