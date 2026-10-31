"""Opt-in graph seam for the experimental trained purpose analyzer.

The injected operator driver owns preparation, GPU scheduling, process bounds
and verified checkpoint loading. This file never starts inference itself.
"""

from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from pathlib import Path

from engine.visual_analysis import bind_analysis

from .adapter_analysis import (
    _identity,
    _purpose_views,
    _ref,
    _source,
    build_adapter_request,
    complete_adapter_analysis,
    digest,
    source_digest,
)

MODEL_KEYS = {"base", "adapter_sha256", "adapter_config_sha256", "code_sha256"}


def create_adapter_roles(
    delegated_roles,
    *,
    context_builder=None,
    score_runner,
    model_identity,
    driver_identity,
    mode="off",
    purpose_views=2,
    view_builder=None,
):
    """Compose a classifier analyzer with existing organizer/verifier roles.

    The returned instance preserves the graph's LocalModelRoles identity seam.
    ``score_runner(bundle, timeout_seconds=...)`` returns exact prediction bytes,
    ``receipt`` and ``prepared_identity`` obtained by trusted preparation BEFORE
    inference. Its prepared model/code pins must match ``model_identity``.
    """
    from .providers import LocalModelRoles

    if mode not in {"off", "diagnostic"}:
        raise ValueError("Unqualified adapter roles support off or diagnostic mode only")
    views = _purpose_views(purpose_views)
    if not isinstance(model_identity, dict) or set(model_identity) != MODEL_KEYS:
        raise ValueError(
            "Adapter role configuration must pin actual base/adapter/config/worker code"
        )
    _identity({**model_identity, "recipe_sha256": "0" * 64, "dataset_sha256": "0" * 64})
    if (
        not callable(score_runner)
        or (views == 2 and (not callable(context_builder) or view_builder is not None))
        or (views == 3 and (not callable(view_builder) or context_builder is not None))
    ):
        raise ValueError(
            "Adapter graph roles require explicit version-matched "
            "operator preparation/scoring callbacks"
        )
    if not isinstance(driver_identity, dict):
        raise ValueError("Adapter scoring driver needs a stable configuration identity")
    digest(driver_identity)

    class AdapterRoles(LocalModelRoles):
        def __init__(self):
            self._delegated = delegated_roles
            self._model = deepcopy(model_identity)
            self._driver_identity = deepcopy(driver_identity)
            self._mode = mode
            self.execution_scope = (
                "supervised_offline_runtime"
                if getattr(delegated_roles, "execution_scope", None) == "supervised_offline_runtime"
                and driver_identity.get("execution_scope") == "supervised_offline_runtime"
                else "injected_adapter_protocol_test"
            )

        def configuration_identity(self):
            code = {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in (
                    "adapter_analysis.py",
                    "adapter_roles.py",
                    "adapter_context.py",
                    "adapter_stream.py",
                )
            }
            value = {
                "schema_version": 1,
                "execution_scope": self.execution_scope,
                "delegated_organizer_verifier": self._delegated.configuration_identity(),
                "analyzer": {
                    "kind": "saved_adapter_purpose_classifier",
                    "mode": self._mode,
                    "purpose_views": views,
                    "model_identity": deepcopy(self._model),
                    "driver_identity": deepcopy(self._driver_identity),
                    "bridge_code_sha256": code,
                    "deployment_enabled": False,
                    "qualified": False,
                },
            }
            value["digest"] = digest(value)
            return value

        def organize(self, *args, **kwargs):
            return self._delegated.organize(*args, **kwargs)

        def verify(self, *args, **kwargs):
            return self._delegated.verify(*args, **kwargs)

        def analyze(self, source, source_render, *, timeout_seconds=None):
            empty = bind_analysis(
                {"observations": [], "abstain": True},
                source_digest=source_digest(source),
                source_render_sha256=_ref(source_render)["sha256"],
                known_element_ids=_source(source),
            )
            fallback = {
                "role": "analyzer",
                "executed": False,
                "valid": False,
                "abstained": True,
                "analysis": empty,
                "trace": {
                    "invoked": False,
                    "deployment_enabled": False,
                    "qualified": False,
                    "experimental_only": True,
                    "mode": self._mode,
                    "purpose_views": views,
                },
            }
            if self._mode == "off":
                fallback["trace"]["fallback_reason"] = "adapter_bridge_disabled"
                return fallback
            try:
                seconds = 180 if timeout_seconds is None else timeout_seconds
                if (
                    type(seconds) not in {int, float}
                    or not math.isfinite(seconds)
                    or not 0 < seconds <= 600
                ):
                    raise ValueError("Adapter graph timeout must be finite and within (0,600]")
                started = time.monotonic()
                source_pin = source_digest(source)
                if views == 2:
                    contexts = context_builder(deepcopy(source), deepcopy(source_render))
                    bundle = build_adapter_request(source, source_render, contexts)
                else:
                    prepared_views = view_builder(deepcopy(source), deepcopy(source_render))
                    if (
                        not isinstance(prepared_views, dict)
                        or not {"context_images", "isolated_images", "isolated_receipts"}
                        <= prepared_views.keys()
                    ):
                        raise ValueError(
                            "Three-view builder must provide actual native evidence maps"
                        )
                    bundle = build_adapter_request(
                        source,
                        source_render,
                        prepared_views["context_images"],
                        purpose_views=3,
                        isolated_images=prepared_views["isolated_images"],
                        isolated_receipts=prepared_views["isolated_receipts"],
                    )
                if not bundle["records"]:
                    fallback["trace"].update(
                        fallback_reason="no_grounded_context_images",
                        initial_abstentions=bundle["initial_abstentions"],
                    )
                    return fallback
                if views == 3:
                    from .adapter_context import verify_source_views

                    verify_source_views(source, source_render, bundle)
                transmitted = deepcopy(bundle)
                fallback["trace"]["invoked"] = True
                remaining = seconds - (time.monotonic() - started) if views == 3 else seconds
                if remaining <= 0:
                    raise ValueError("Adapter graph deadline expired while preparing native views")
                scored = score_runner(transmitted, timeout_seconds=remaining)
                if transmitted != bundle or source_digest(source) != source_pin:
                    raise ValueError("Adapter scoring mutated its source/request binding")
                if not isinstance(scored, dict) or set(scored) != {
                    "prediction_bytes",
                    "receipt",
                    "prepared_identity",
                }:
                    raise ValueError(
                        "Adapter driver must return scores, receipt and "
                        "pre-inference prepared identity"
                    )
                prepared = _identity(scored["prepared_identity"])
                if any(prepared[key] != value for key, value in self._model.items()):
                    raise ValueError(
                        "Adapter driver changed the operator-pinned model/code identity"
                    )
                result = complete_adapter_analysis(
                    source,
                    source_render,
                    bundle,
                    scored["prediction_bytes"],
                    scored["receipt"],
                    expected_identity=prepared,
                    mode=self._mode,
                )
                fallback = result
                if views == 3:
                    verify_source_views(source, source_render, bundle)
                result["trace"].update(
                    driver_identity=deepcopy(self._driver_identity),
                    request_sha256=bundle["request_sha256"],
                )
                if time.monotonic() - started > seconds:
                    result.update(valid=False, abstained=True, analysis=empty)
                    result["trace"]["fallback_reason"] = "adapter_analyzer_graph_deadline"
                return result
            except (
                ValueError,
                OSError,
                KeyError,
                TypeError,
                UnicodeError,
                RecursionError,
                RuntimeError,
            ) as exc:
                fallback.update(valid=False, abstained=True, analysis=empty)
                fallback["trace"]["fallback_reason"] = str(exc)[:600]
                return fallback

    return AdapterRoles()
