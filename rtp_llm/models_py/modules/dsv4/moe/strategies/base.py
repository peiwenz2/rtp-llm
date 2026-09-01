"""Routed-expert strategy interface + registry.

A *strategy* owns the per-rank routed-expert compute. The MoE layer drives
``Gate`` (token → expert routing) and, normally, the *shared* expert; a fused
strategy may own the shared expert too and return ``routed + shared``.

The framework is intentionally NOT involved here — see
``.claude/plans/optimized-riding-mist.md`` for why we keep this dsv4-internal
rather than going through ``rtp_llm.models_py.modules.factory.fused_moe``.

Strategies (priority high→low for ``forced=None``):

    ep_size  model / kernel               → strategy
    --------------------------------------------------------
    >1       shared experts present        MegaMoESEStrategy
    >1       no shared experts             MegaMoEStrategy
    >1       selected Mega unavailable     RuntimeError
    1        grouped FP4 kernel available  GroupedFP4Strategy
    1        grouped unavailable           LocalLoopStrategy

A model can override the auto-pick through the ``MoE(strategy=...)``
constructor argument. Production threads the public ``--moe_strategy`` value
through ``MoeConfig -> V4Args -> Block -> MoE``. Selection environment toggles
are intentionally not read here, so an explicit argument remains authoritative.
"""

from __future__ import annotations

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


def register_strategy(cls: Type[RoutedExpertsStrategy]) -> Type[RoutedExpertsStrategy]:
    """Decorator: append ``cls`` to ``_STRATEGY_PRIORITY``.

    Order of import = order of priority. Convention: strategies/__init__.py
    imports them in priority order high→low.
    """
    if cls not in _STRATEGY_PRIORITY:
        _STRATEGY_PRIORITY.append(cls)
    return cls


def _resolve_forced(strategy_arg: Optional[str]) -> tuple[Optional[str], bool]:
    """Normalize the public constructor/CLI value.

    ``None``, an empty string and ``auto`` request model-aware automatic
    selection. Every named strategy is explicit and therefore strict.
    """
    if strategy_arg is None:
        return None, False
    strategy = str(strategy_arg).strip()
    if not strategy or strategy == "auto":
        return None, False
    return strategy, True


def select_strategy(
    cfg: MoeCfg,
    forced: Optional[str] = None,
    strict: bool = True,
) -> Type[RoutedExpertsStrategy]:
    """Pick a strategy class for ``cfg``.

    ``forced`` is a public strategy name. Named selections are strict in the
    production path; ``strict=False`` remains only for direct internal callers.
    """
    if forced is not None:
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
        default_name = "mega_moe_se" if cfg.n_shared_experts > 0 else "mega_moe"
        mega_cls = next((c for c in _STRATEGY_PRIORITY if c.name == default_name), None)
        if mega_cls is None:
            raise RuntimeError(
                f"DSV4 EP MoE default strategy {default_name!r} is not registered."
            )
        if mega_cls.can_handle(cfg):
            return mega_cls
        if default_name == "mega_moe_se":
            from rtp_llm.models_py.modules.dsv4.moe.mega_se_buf import (
                _mega_moe_se_unavailable_reason,
            )

            reason = _mega_moe_se_unavailable_reason()
        else:
            from rtp_llm.models_py.modules.dsv4.moe.mega_buf import (
                _mega_moe_unavailable_reason,
            )

            reason = _mega_moe_unavailable_reason()

        raise RuntimeError(
            f"DSV4 EP MoE selected {default_name!r} from model metadata, but "
            "that strategy is unavailable; fallback is disabled. "
            f"layer_id={cfg.layer_id}, ep_size={cfg.ep_size}. "
            f"Reason: {reason or 'unknown availability failure'}."
        )

    for cls in _STRATEGY_PRIORITY:
        if cls.can_handle(cfg):
            return cls
    raise RuntimeError(
        f"No MoE strategy can handle cfg (layer_id={cfg.layer_id}, "
        f"ep_size={cfg.ep_size})"
    )
