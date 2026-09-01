"""Routed-expert strategy interface + registry.

A *strategy* owns the per-rank routed-expert compute. The MoE layer drives
``Gate`` (token → expert routing) and, normally, the *shared* expert; a fused
strategy may own the shared expert too and return ``routed + shared``.

The framework is intentionally NOT involved here — see
``.claude/plans/optimized-riding-mist.md`` for why we keep this dsv4-internal
rather than going through ``rtp_llm.models_py.modules.factory.fused_moe``.

Strategies (priority high→low for automatic selection):

    ep_size  model / kernel               → strategy
    --------------------------------------------------------
    >1       shared experts + SE available MegaMoESEStrategy
    >1       shared experts + SE missing   MegaMoEStrategy + standalone shared
    >1       no shared experts             MegaMoEStrategy
    >1       base Mega unavailable         RuntimeError
    1        grouped FP4 kernel available  GroupedFP4Strategy
    1        grouped unavailable           LocalLoopStrategy

A model can override the auto-pick through the ``MoE(strategy=...)``
constructor argument. Production threads the public ``--moe_strategy`` value
through ``MoeConfig -> V4Args -> Block -> MoE``. Legacy DSV4 environment
toggles are translated for one compatibility cycle, with deprecation warnings
and explicit conflict detection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Type

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MoeCfg:
    """Per-layer MoE configuration shared across all strategies.

    Frozen because strategies cache stuff keyed off it; mutating after
    construction would silently invalidate those caches.
    """

    layer_id: int
    dim: int
    moe_inter_dim: int
    n_routed_experts: int
    n_activated_experts: int  # topk
    n_shared_experts: int
    swiglu_limit: float
    ep_size: int
    ep_rank: int
    n_local_experts: int
    local_expert_start: int
    local_expert_end: int
    max_tokens_per_rank: int


class RoutedExpertsStrategy(nn.Module):
    """Single-card or multi-card routed-expert compute.

    Inherits ``nn.Module`` so that strategies (notably ``LocalLoopStrategy``)
    can hold ``nn.ModuleList`` of ``Expert`` children whose Parameters propagate
    correctly through ``MoE.to(device)`` / state_dict traversal.

    The MoE layer is normally responsible for:
      - ``Gate`` (routing scores + topk)
      - the shared expert (one ``Expert`` instance)
      - dispatching to the chosen strategy

    A strategy with ``routed_includes_shared=True`` instead owns the shared
    expert weights and returns the combined result.

    A strategy is responsible for:
      - holding its own slice of routed-expert weights (loaded in ``setup_weights``)
      - producing ``[N, D] fp32`` per-token routed-sum from
        ``(x: [N, D] BF16, weights: [N, topk] FP32, indices: [N, topk] int64)``

    A strategy MUST handle cuda-graph capture state internally (e.g.
    ``LocalLoopStrategy.forward`` checks ``torch.cuda.is_current_stream_capturing()``
    and dispatches to a graph-safe variant). The MoE layer does NOT switch
    strategies based on capture state.

    Subclasses MUST call ``super().__init__()`` first so nn.Module bookkeeping
    is initialised. They override ``forward`` directly (it doubles as both
    nn.Module's forward hook and the strategy interface contract) and must
    define ``setup_weights`` + ``can_handle``.
    """

    # Registered names currently include mega_moe, mega_moe_se,
    # grouped_fp4, local_loop, and deepep.
    name: ClassVar[str]

    # True when ``forward`` already returns ``routed + shared`` (the strategy
    # fuses the shared expert internally). The ``MoE`` layer then skips its own
    # shared-expert executor and the ``combine_routed_and_shared`` add. Only
    # Mega variants that fuse the shared expert set this True.
    routed_includes_shared: ClassVar[bool] = False

    def __init__(self, cfg: MoeCfg):
        super().__init__()
        self.cfg = cfg

    def setup_weights(self, layer_weights: Dict) -> None:
        """Pop the strategy's own routed-expert stacks from ``layer_weights``
        (the framework's per-layer ``ModelWeights.weights[layer_id]`` dict
        keyed by ``W.v4_*`` enum). The stacks are already EP-sliced by the
        loader: each ``W.v4_routed_w{1,2,3}_{w,s}`` has shape ``[E_local, ...]``.

        Each strategy's docstring lists the exact W keys it pops, so a
        post-init audit can detect leftover keys (= bug).
        """
        raise NotImplementedError

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,  # [N, D] BF16
        weights: torch.Tensor,  # [N, topk] FP32
        indices: torch.Tensor,  # [N, topk] int64 GLOBAL expert id
    ) -> torch.Tensor:  # [N, D] FP32
        """Route + compute. Returns per-token routed-expert sum in fp32."""
        raise NotImplementedError

    def can_use_gate_pack_static(self, gate) -> bool:
        """Whether this strategy can use the MegaMoE gate-pack fast path.

        The default strategy contract is "not supported"; Mega strategies
        override it after checking env/static model properties.
        """
        return False

    def forward_with_gate_pack(
        self,
        x: torch.Tensor,
        gate,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """Optional fast path that fuses router gate + MegaMoE input packing."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support MegaMoE gate-pack"
        )

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        """Whether this strategy is applicable for ``cfg`` in the current
        runtime (env vars, kernel availability, dist init, SM arch, ...).

        Does NOT check cuda-graph capture state — that is forward's concern.
        """
        raise NotImplementedError


# --- selection -------------------------------------------------------------

# All known strategies, in priority order. Populated by ``register_strategy``
# from each strategy module's import side-effect (see strategies/__init__.py
# — importing a strategy class registers it).
_STRATEGY_PRIORITY: list[Type[RoutedExpertsStrategy]] = []
_MEGA_SE_FALLBACK_WARNED = False
_DEPRECATION_WARNED: set[str] = set()

_LEGACY_STRATEGY_ALIASES = {
    "mega": "mega_moe",
    "mega_se": "mega_moe_se",
}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _warn_deprecated_once(key: str, message: str, *args) -> None:
    if key in _DEPRECATION_WARNED:
        return
    logging.warning(message, *args)
    _DEPRECATION_WARNED.add(key)


def register_strategy(cls: Type[RoutedExpertsStrategy]) -> Type[RoutedExpertsStrategy]:
    """Decorator: append ``cls`` to ``_STRATEGY_PRIORITY``.

    Order of import = order of priority. Convention: strategies/__init__.py
    imports them in priority order high→low.
    """
    if cls not in _STRATEGY_PRIORITY:
        _STRATEGY_PRIORITY.append(cls)
    return cls


def _normalize_strategy_name(value: str, source: str) -> str:
    strategy = str(value).strip()
    if strategy == "mega_fused":
        raise RuntimeError(
            f"MoE strategy 'mega_fused' from {source} is no longer supported. "
            "Use MOE_STRATEGY=mega_moe_se for fused shared-expert execution."
        )
    normalized = _LEGACY_STRATEGY_ALIASES.get(strategy, strategy)
    if normalized != strategy:
        _warn_deprecated_once(
            f"strategy:{strategy}",
            "MoE strategy name %r from %s is deprecated; use %r",
            strategy,
            source,
            normalized,
        )
    return normalized


def _read_legacy_flag(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    _warn_deprecated_once(
        f"env:{name}",
        "%s is deprecated and will be removed after one compatibility "
        "cycle; use MOE_STRATEGY instead",
        name,
    )
    value = raw.strip().lower()
    if value == "auto":
        return None
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"Invalid legacy MoE toggle {name}={raw!r}; expected one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES | {'auto'})}"
    )


def _resolve_forced(
    strategy_arg: Optional[str],
    *,
    ep_size: Optional[int] = None,
) -> tuple[Optional[str], bool]:
    """Merge the public strategy value with one-cycle legacy compatibility.

    The public constructor value (normally populated from ``MOE_STRATEGY``) is
    authoritative unless a deprecated negative toggle explicitly disables that
    backend. Legacy variables remain functional but emit deprecation warnings;
    if old and new sources request different strategies, startup fails instead
    of silently changing the execution backend. Positive historical toggles are
    best-effort hints except for the explicit Mega-SE opt-in. Negative toggles
    remain strict rollback constraints, scoped to the topology where the
    backend is applicable: Mega toggles affect EP configurations and the
    grouped toggle affects single-rank execution.
    """
    public_strategy: Optional[str] = None
    if strategy_arg is not None:
        raw_public = str(strategy_arg).strip()
        if raw_public and raw_public != "auto":
            public_strategy = _normalize_strategy_name(
                raw_public, "MoE strategy argument"
            )

    legacy_candidates: list[tuple[str, str, bool]] = []
    mega_topology = ep_size is None or ep_size > 1
    grouped_topology = ep_size is None or ep_size == 1
    legacy_strategy = os.environ.get("DSV4_MOE_STRATEGY")
    if legacy_strategy is not None:
        _warn_deprecated_once(
            "env:DSV4_MOE_STRATEGY",
            "DSV4_MOE_STRATEGY is deprecated; use MOE_STRATEGY instead",
        )
        raw_legacy = legacy_strategy.strip()
        if raw_legacy and raw_legacy != "auto":
            legacy_candidates.append(
                (
                    "DSV4_MOE_STRATEGY",
                    _normalize_strategy_name(raw_legacy, "DSV4_MOE_STRATEGY"),
                    True,
                )
            )

    use_mega = _read_legacy_flag("DSV4_USE_MEGA_MOE")
    use_mega_se = _read_legacy_flag("DSV4_USE_MEGA_MOE_SE")
    use_mega_fused = _read_legacy_flag("DSV4_USE_MEGA_MOE_FUSED")
    use_grouped = _read_legacy_flag("DSV4_USE_GROUPED_FP4")

    if use_mega_fused is True:
        # There are currently no deployments using the removed fused backend,
        # so deleting that implementation has no compatibility impact. Keep
        # the legacy flag only to provide an actionable migration error.
        raise RuntimeError(
            "DSV4_USE_MEGA_MOE_FUSED=1 requests the removed mega_fused "
            "backend. Unset the legacy variable and use "
            "MOE_STRATEGY=mega_moe_se instead."
        )

    # Preserve the old negative rollback contracts for one compatibility cycle.
    # DSV4 has no supported non-Mega EP backend, so disabling the Mega family is
    # an immediate configuration error. Disabling grouped FP4 selects the local
    # implementation; disabling Mega-SE selects routed-only Mega plus the
    # standalone shared expert. The old fused toggle also had a strict false
    # branch: it must keep the non-fused routed-only Mega implementation.
    if mega_topology and use_mega is False:
        raise RuntimeError(
            "DSV4_USE_MEGA_MOE=0 disables Mega MoE, but DeepSeek-V4 EP "
            f"requires the Mega family (ep_size={ep_size!r}). Remove the "
            "deprecated variable to use MOE_STRATEGY, or run with ep_size=1."
        )
    if mega_topology and use_mega_se is False:
        legacy_candidates.append(("DSV4_USE_MEGA_MOE_SE=0", "mega_moe", True))
    if mega_topology and use_mega_fused is False:
        legacy_candidates.append(("DSV4_USE_MEGA_MOE_FUSED=0", "mega_moe", True))
    if grouped_topology and use_grouped is False:
        legacy_candidates.append(("DSV4_USE_GROUPED_FP4=0", "local_loop", True))

    # A generic Mega hint is subsumed by the more specific SE opt-in.
    if mega_topology and use_mega is True and use_mega_se is not True:
        legacy_candidates.append(("DSV4_USE_MEGA_MOE", "mega_moe", False))
    if use_mega_se is True:
        legacy_candidates.append(("DSV4_USE_MEGA_MOE_SE", "mega_moe_se", True))
    if grouped_topology and use_grouped is True:
        legacy_candidates.append(("DSV4_USE_GROUPED_FP4", "grouped_fp4", False))

    legacy_names = {candidate[1] for candidate in legacy_candidates}
    if len(legacy_names) > 1:
        detail = ", ".join(f"{source}->{name}" for source, name, _ in legacy_candidates)
        raise RuntimeError(
            f"Conflicting legacy MoE configuration: {detail}. Use "
            "MOE_STRATEGY=<name> as the single source of truth."
        )

    legacy_name = next(iter(legacy_names), None)
    if public_strategy is not None and legacy_name is not None:
        if public_strategy != legacy_name:
            raise RuntimeError(
                "Conflicting MoE configuration: "
                f"MOE_STRATEGY={public_strategy!r} but legacy variables select "
                f"{legacy_name!r}. Remove the legacy variables."
            )
        return public_strategy, True
    if public_strategy is not None:
        return public_strategy, True
    if legacy_name is not None:
        strict = any(candidate[2] for candidate in legacy_candidates)
        return legacy_name, strict
    return None, False


def select_strategy(
    cfg: MoeCfg,
    forced: Optional[str] = None,
    strict: bool = True,
) -> Type[RoutedExpertsStrategy]:
    """Pick a strategy class for ``cfg``.

    ``forced`` is a public strategy name. Named selections are strict in the
    production path; ``strict=False`` remains only for direct internal callers.
    """
    global _MEGA_SE_FALLBACK_WARNED

    if forced is not None:
        if forced == "mega_moe_se" and cfg.n_shared_experts != 1:
            raise RuntimeError(
                "Forced MoE strategy 'mega_moe_se' requires exactly one shared "
                f"expert, got n_shared_experts={cfg.n_shared_experts} "
                f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
            )
        for cls in _STRATEGY_PRIORITY:
            if cls.name == forced:
                if cls.can_handle(cfg):
                    if cfg.ep_size > 1 and cls.name not in (
                        "mega_moe",
                        "mega_moe_se",
                    ):
                        raise RuntimeError(
                            "DSV4 EP MoE requires MegaMoEStrategy. "
                            f"Requested strategy {forced!r} would bypass Mega "
                            f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size})."
                        )
                    return cls
                if strict:
                    raise RuntimeError(
                        f"Forced MoE strategy {forced!r} cannot handle cfg "
                        f"(layer_id={cfg.layer_id}, ep_size={cfg.ep_size}). "
                        "Check runtime and kernel availability."
                    )
                # Non-strict direct callers fall through to auto-pick.
                break
        else:
            names = [c.name for c in _STRATEGY_PRIORITY]
            raise RuntimeError(f"Unknown MoE strategy {forced!r}. Available: {names}")

    if cfg.ep_size > 1:
        candidate_names = (
            ["mega_moe_se", "mega_moe"] if cfg.n_shared_experts > 0 else ["mega_moe"]
        )
        failures: list[str] = []
        for candidate_name in candidate_names:
            mega_cls = next(
                (c for c in _STRATEGY_PRIORITY if c.name == candidate_name), None
            )
            if mega_cls is None:
                failures.append(f"{candidate_name}: not registered")
                continue
            if mega_cls.can_handle(cfg):
                if candidate_name == "mega_moe" and len(candidate_names) > 1:
                    from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
                        _mega_moe_se_unavailable_reason,
                    )

                    if not _MEGA_SE_FALLBACK_WARNED:
                        logging.warning(
                            "DSV4 auto strategy cannot use mega_moe_se (%s); "
                            "falling back to mega_moe with the standalone "
                            "shared expert",
                            _mega_moe_se_unavailable_reason()
                            or "unknown availability failure",
                        )
                        _MEGA_SE_FALLBACK_WARNED = True
                return mega_cls
            if candidate_name == "mega_moe_se":
                from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
                    _mega_moe_se_unavailable_reason,
                )

                reason = _mega_moe_se_unavailable_reason()
            else:
                from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
                    _mega_moe_unavailable_reason,
                )

                reason = _mega_moe_unavailable_reason()
            failures.append(
                f"{candidate_name}: {reason or 'unknown availability failure'}"
            )

        raise RuntimeError(
            "DSV4 EP MoE has no available Mega strategy after trying "
            f"{candidate_names!r}; layer_id={cfg.layer_id}, "
            f"ep_size={cfg.ep_size}. Reasons: {'; '.join(failures)}."
        )

    for cls in _STRATEGY_PRIORITY:
        if cls.can_handle(cfg):
            return cls
    raise RuntimeError(
        f"No MoE strategy can handle cfg (layer_id={cfg.layer_id}, "
        f"ep_size={cfg.ep_size})"
    )
