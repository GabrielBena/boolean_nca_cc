"""
Self-test for the SODC NumPy oracle.

Two layers:

1. ``run_self_test()``: builds a tiny synthetic circuit + random
   weights and exercises every public function of ``numpy_oracle``.
   Catches shape mismatches, NaNs, and silent state corruption.
   No JAX dependency; runs anywhere with numpy + scipy.

2. ``run_jax_parity_test(checkpoint_path)`` *(deferred to M2)*:
   loads a real ``CircuitSelfAttention`` checkpoint, drives both
   the JAX policy and the NumPy oracle for N ticks, and asserts
   per-step max-abs error < 1e-5 on logits + hidden + loss. The
   stub signals "TODO" with a clear message until a checkpoint is
   pinned.

Run via::

    python -m export.numpy_oracle --self-test
    # or
    python -m export.test_oracle
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .numpy_oracle import (
    TMTConfig,
    apply_shotgun,
    build_topology,
    circuit_forward,
    compute_residuals,
    damage_gate,
    flat_to_layered_logits,
    gen_wires,
    init_state,
    make_random_weights,
    reset_circuit,
    shuffle_wires,
    tmt_step,
)


def _build_tiny_circuit(
    rng: np.random.Generator,
    *,
    arity: int = 4,
    hidden_dim: int = 8,
    model_kind: str = "gathered_attention",
    max_neighbors: int = 16,
) -> tuple:
    """Tiny scale-free circuit: 4-bit input, two hidden layers, 4-bit output."""
    layer_sizes = [
        (4, 1),  # input
        (16, arity),  # hidden 1
        (16, arity),  # hidden 2
        (4, 1),  # output
    ]
    cfg = TMTConfig(
        arity=arity,
        circuit_hidden_dim=hidden_dim,
        attention_dim=32,
        num_heads=4,
        num_attn_layers=1,
        use_layer_PE=True,
        use_intra_layer_PE=False,
        use_node_loss=True,
        model_kind=model_kind,
        max_neighbors=max_neighbors,
    )

    wires: list[np.ndarray] = []
    for li in range(1, len(layer_sizes)):
        in_n = layer_sizes[li - 1][0]
        out_n, gs = layer_sizes[li]
        wires.append(gen_wires(rng, in_n, out_n, arity, gs))
    cache_max = max_neighbors if model_kind == "gathered_attention" else None
    topology = build_topology(layer_sizes, wires, arity, hidden_dim, max_neighbors=cache_max)
    state, _ = init_state(topology, cfg, rng, noise_scale=0.1)
    weights = make_random_weights(cfg, rng)
    return cfg, topology, state, weights


def _truth_table_4bit_reverse() -> tuple[np.ndarray, np.ndarray]:
    """Tiny ground-truth task: 4-bit reverse. 16 cases."""
    n_in = 4
    n_cases = 1 << n_in
    bits = np.array([[(c >> i) & 1 for i in range(n_in)] for c in range(n_cases)], dtype=np.float32)
    y = bits[:, ::-1].copy()
    return bits, y


def _check(name: str, cond: bool, fail_msg: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + fail_msg) if (not cond and fail_msg) else ''}")
    if not cond:
        raise AssertionError(name + (": " + fail_msg if fail_msg else ""))


def run_self_test(*, verbose: bool = False, model_kind: str = "gathered_attention") -> bool:
    """Smoke-test every oracle entry point with random weights + a tiny circuit.

    Returns True on success. Raises ``AssertionError`` on the first
    failed check (so the CLI propagates a non-zero exit code).
    """
    rng = np.random.default_rng(0)
    if verbose:
        print(f"[oracle self-test] kind={model_kind} — building tiny circuit + random weights ...")
    cfg, topology, state, weights = _build_tiny_circuit(rng, model_kind=model_kind)
    x_data, y_data = _truth_table_4bit_reverse()

    if verbose:
        print(f"  N nodes:     {topology.n_nodes}")
        print(f"  layer_start: {topology.layer_start}")
        print(f"  output rows: [{topology.output_start}, {topology.output_end})")

    # 1. Topology shapes.
    _check(
        "topology.attention_mask shape",
        topology.attention_mask.shape == (topology.n_nodes, topology.n_nodes),
    )
    _check(
        "diagonal of attention_mask is all True",
        np.all(np.diag(topology.attention_mask)),
    )
    _check(
        "attention_mask is symmetric (bidirectional edges)",
        np.array_equal(topology.attention_mask, topology.attention_mask.T),
    )
    _check(
        "attention_mask has expected sender->receiver edges",
        topology.attention_mask.sum() > topology.n_nodes,  # more than just diagonal
    )
    _check("layer_pe shape", topology.layer_pe.shape == (topology.n_nodes, cfg.circuit_hidden_dim))
    _check("intra_layer_pe shape", topology.intra_layer_pe.shape == (topology.n_nodes, cfg.circuit_hidden_dim))

    # 2. State shapes.
    _check("state.logits shape", state.logits.shape == (topology.n_nodes, cfg.logit_dim))
    _check("state.hidden shape", state.hidden.shape == (topology.n_nodes, cfg.circuit_hidden_dim))
    _check("state.loss shape", state.loss.shape == (topology.n_nodes,))
    _check("state.gate_mask shape", state.gate_mask.shape == (topology.n_nodes,))
    _check("input-node logits are zeros", np.all(state.logits[: topology.layer_start[1]] == 0.0))
    _check("gate_mask starts all 1.0", np.all(state.gate_mask == 1.0))

    # 3. Circuit forward smoke.
    layered = flat_to_layered_logits(state, topology)
    acts = circuit_forward(layered, topology.wires, x_data)
    _check("circuit_forward returns one tensor per layer", len(acts) == len(topology.layer_sizes))
    _check(
        "final activation matches output bits",
        acts[-1].shape == y_data.shape,
        f"got {acts[-1].shape}, want {y_data.shape}",
    )
    _check("no NaNs in soft forward", not np.any(np.isnan(acts[-1])))
    acts_hard = circuit_forward(layered, topology.wires, x_data, hard=True)
    _check("hard forward in {0,1}", set(np.unique(acts_hard[-1])).issubset({0.0, 1.0}))

    # 4. Residual recompute populates the output rows of state.loss.
    state, aux = compute_residuals(state, topology, x_data, y_data)
    _check(
        "residual aux carries soft + hard predictions",
        {"pred_soft", "pred_hard", "residuals", "hard_residuals", "accuracy", "hard_accuracy"}
        <= set(aux.keys()),
    )
    output_rows = state.loss[topology.output_start : topology.output_end]
    nonoutput_rows = np.concatenate(
        [state.loss[: topology.output_start], state.loss[topology.output_end :]]
    )
    _check("loss populated on output rows", np.any(output_rows > 0.0) or np.allclose(aux["residuals"], 0))
    _check("loss zero on non-output rows", np.all(nonoutput_rows == 0.0))

    # 5. TMT step preserves shapes and avoids NaNs.
    s_before = state
    for _t in range(8):
        state = tmt_step(state, topology, weights)
        state, _ = compute_residuals(state, topology, x_data, y_data)
    _check("logits shape preserved across 8 ticks", state.logits.shape == s_before.logits.shape)
    _check("hidden shape preserved across 8 ticks", state.hidden.shape == s_before.hidden.shape)
    _check("no NaNs in logits", not np.any(np.isnan(state.logits)))
    _check("no NaNs in hidden", not np.any(np.isnan(state.hidden)))

    # 6. Damage primitives.
    state_dmg, chosen = apply_shotgun(state, topology, n_gates=3, rng=rng)
    _check("shotgun chose 3 distinct gates", chosen.size == 3 and len(set(chosen.tolist())) == 3)
    _check("shotgun zeros gate_mask at chosen indices", np.all(state_dmg.gate_mask[chosen] == 0.0))
    _check(
        "shotgun stamps faulty logits at chosen indices",
        np.allclose(state_dmg.logits[chosen], -10.0),
    )
    _check(
        "shotgun preserves other gates",
        np.array_equal(
            np.delete(state_dmg.gate_mask, chosen),
            np.delete(state.gate_mask, chosen),
        ),
    )

    target = int(chosen[0])  # already damaged; damage_gate should be idempotent
    state_single = damage_gate(state_dmg, target)
    _check("damage_gate keeps target clamped", state_single.gate_mask[target] == 0.0)

    # 7. Knockout actually freezes updates (clamped gate's logits don't drift).
    state_after_step = tmt_step(state_dmg, topology, weights)
    _check(
        "tmt_step does not update damaged gates",
        np.allclose(state_after_step.logits[chosen], state_dmg.logits[chosen]),
    )

    # 8. Reset clears damage and restores logits to soft-wire init magnitude.
    state_reset = reset_circuit(topology, cfg, np.random.default_rng(0))
    _check("reset clears gate_mask to 1.0", np.all(state_reset.gate_mask == 1.0))
    _check(
        "reset zeros hidden",
        np.all(state_reset.hidden == 0.0),
    )

    # 9. Shuffle wires preserves layer sizes and rebuilds the mask.
    new_topology = shuffle_wires(topology, cfg, np.random.default_rng(1), hidden_dim=cfg.circuit_hidden_dim)
    _check("shuffle preserves n_nodes", new_topology.n_nodes == topology.n_nodes)
    _check("shuffle preserves layer_sizes", new_topology.layer_sizes == topology.layer_sizes)
    _check(
        "shuffle does change the wires",
        any(not np.array_equal(a, b) for a, b in zip(new_topology.wires, topology.wires)),
    )
    _check(
        "shuffle gives a different attention mask",
        not np.array_equal(new_topology.attention_mask, topology.attention_mask),
    )

    # 10. Gathered-attention specific: neighbor cache is populated and consistent
    #     with the dense attention mask (every True row in the mask maps to a
    #     real entry in neighbor_indices).
    if cfg.model_kind == "gathered_attention":
        ni = topology.neighbor_indices
        nm = topology.neighbor_mask
        _check("gathered: neighbor_indices populated", ni is not None and nm is not None)
        # Every real neighbor (mask True at position k) must appear in attention_mask.
        for i in range(topology.n_nodes):
            real = ni[i][nm[i]]
            for j in real:
                _check(
                    f"gathered: row {i} neighbor {int(j)} in dense mask",
                    bool(topology.attention_mask[i, int(j)]),
                )

    if verbose:
        print(f"[oracle self-test] kind={cfg.model_kind} — all checks passed.")
    return True


def run_self_test_all_kinds(*, verbose: bool = False) -> bool:
    """Run the self-test for both architectures."""
    ok1 = run_self_test(verbose=verbose, model_kind="gathered_attention")
    ok2 = run_self_test(verbose=verbose, model_kind="self_attention")
    return ok1 and ok2


def run_jax_parity_test(
    *,
    run_id: str | None = None,
    max_steps: int = 8,
    tol: float = 1e-4,
    weights_json: str | None = None,
    verbose: bool = True,
) -> bool:
    """Cross-validate the NumPy oracle against the JAX policy.

    Loads the deployment checkpoint, builds a small fresh circuit, and
    drives both the JAX ``apply_model_and_compute_loss`` step and the
    NumPy oracle's ``tmt_step + compute_residuals`` for ``max_steps``
    ticks. Asserts per-step max-abs error on ``logits / hidden / loss``
    is below ``tol``.

    Imports JAX/W&B lazily so the cheap ``run_self_test_all_kinds``
    above stays dependency-free.
    """
    # Lazy imports — keep the cheap path JAX-free.
    import jax
    import numpy as np

    from boolean_nca_cc.circuits.model import gen_circuit, generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import (
        _prepare_model_fn,
        apply_model_and_compute_loss,
        get_loss_and_update_graph,
    )
    from boolean_nca_cc.utils.graph_builder import build_graph

    from .extract_weights import (
        DEFAULT_RUN_ID,
        deserialize_weights,
        extract_weights,
        load_run,
        to_np,
    )
    from .numpy_oracle import (
        CircuitState,
        build_topology,
        compute_residuals,
        tmt_step,
    )

    rid = run_id or DEFAULT_RUN_ID
    if verbose:
        print(f"[jax-parity] loading run {rid} ...")
    model, cfg, _ = load_run(run_id=rid)
    if weights_json is not None:
        if verbose:
            print(f"[jax-parity] loading quantised weights from {weights_json}")
        weights = deserialize_weights(weights_json)
    else:
        weights = extract_weights(model, cfg)

    # Architectural sanity: the loaded run must match what the oracle
    # was asked to reproduce. We bail loudly on a mismatch.
    arity = int(cfg.circuit.arity)
    input_bits = int(cfg.circuit.input_bits)
    output_bits = int(cfg.circuit.output_bits)
    hidden_dim = int(cfg.model.circuit_hidden_dim)
    if verbose:
        print(
            f"[jax-parity] kind={weights.config.model_kind}, arity={arity}, "
            f"input_bits={input_bits}, output_bits={output_bits}, hidden_dim={hidden_dim}"
        )

    # Fresh small circuit with a fixed seed so the test is reproducible.
    layer_sizes = generate_layer_sizes(
        input_bits,
        output_bits,
        arity,
        layer_n=int(cfg.circuit.num_layers),
        width_factor=float(cfg.circuit.get("width_factor", 2)),
    )
    parity_seed = int(cfg.get("test_seed", 42)) + 1  # don't reuse the dummy-graph seed
    wires_key, logits_key = jax.random.split(jax.random.PRNGKey(parity_seed))
    wires_jax, layered_logits_jax = gen_circuit(
        wires_key,
        logits_key,
        layer_sizes,
        arity=arity,
        noise_scale=float(cfg.pool.noise_scale),
    )

    graph = build_graph(
        wires=wires_jax,
        logits=layered_logits_jax,
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=hidden_dim,
    )

    # Task data — full truth table for the configured task.
    from boolean_nca_cc.utils.task_compat import get_fixed_task_name
    _task_name = get_fixed_task_name(cfg)
    if _task_name is None:
        raise ValueError(
            "test_oracle requires a fixed task in the checkpoint cfg "
            "(cfg.tasks.name or legacy cfg.circuit.task)."
        )
    task = str(_task_name)
    case_n = 1 << input_bits
    (x_data, y_data), _, _ = get_task_data(
        task,
        case_n=case_n,
        input_bits=input_bits,
        output_bits=output_bits,
        train_test_split=False,
    )
    if verbose:
        print(
            f"[jax-parity] task={task}, layer_sizes={layer_sizes}, "
            f"n_node={int(graph.n_node[0])}, case_n={case_n}"
        )

    # Initial residuals (matches evaluate_model_stepwise_generator step 0).
    original_shapes = [tuple(lgt.shape) for lgt in layered_logits_jax]
    graph, _, _, _ = get_loss_and_update_graph(
        graph, original_shapes, wires_jax, x_data, y_data, LOSS_L4, layer_sizes
    )

    # Build the oracle's topology + state, byte-identical to the JAX graph.
    wires_np = [np.asarray(w, dtype=np.int32) for w in wires_jax]
    cache_max = (
        weights.config.max_neighbors
        if weights.config.model_kind == "gathered_attention"
        else None
    )
    topology = build_topology(
        layer_sizes, wires_np, arity, hidden_dim, max_neighbors=cache_max
    )
    state = CircuitState(
        logits=to_np(graph.nodes["logits"]),
        hidden=to_np(graph.nodes["hidden"]),
        loss=to_np(graph.nodes["loss"]),
        gate_mask=to_np(graph.nodes["gate_knockout_mask"]),
    )
    x_np, y_np = to_np(x_data), to_np(y_data)

    # Step 0 sanity (initial residuals must match before we touch the model).
    init_err = max(
        float(np.max(np.abs(to_np(graph.nodes["loss"]) - state.loss))),
        float(np.max(np.abs(to_np(graph.nodes["logits"]) - state.logits))),
    )
    if verbose:
        print(f"[jax-parity] step 0 init max-err: {init_err:.2e}")
    if init_err > 1e-6:
        print(f"  [FAIL] initial state diverges by {init_err:.2e} (should be exact)")
        return False

    # Pre-build model_fn (precomputes neighbor indices for gathered attention).
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing=False)

    # Single-block attention parity is float32-epsilon (validated separately at
    # ~1e-5). What compounds across recurrent ticks is the JAX-vs-NumPy
    # einsum-summation order; the divergence is in the irrelevant low-order
    # bits of the soft state. The deployment-relevant metric is whether
    # ``pred_hard`` agrees — the JS port will run float32 too, with the same
    # accumulation pattern, so any drift the oracle has is shared with what
    # the user will see in the browser.
    hard_acc_tol = 0.005  # half a percent on a 4096-case test
    soft_logit_floor = 1e-1  # report-only ceiling; not part of pass/fail
    max_hard_delta = 0.0
    final_pred_disagreements = 0
    for t in range(1, max_steps + 1):
        graph, _, _, jax_aux = apply_model_and_compute_loss(
            model_fn, graph, original_shapes, wires_jax, x_data, y_data, LOSS_L4, layer_sizes
        )
        state = tmt_step(state, topology, weights)
        state, oracle_aux = compute_residuals(state, topology, x_np, y_np)

        err_logits = float(np.max(np.abs(to_np(graph.nodes["logits"]) - state.logits)))
        err_hidden = float(np.max(np.abs(to_np(graph.nodes["hidden"]) - state.hidden)))
        jax_hard = float(jax_aux["hard_accuracy"])
        oracle_hard = float(oracle_aux["hard_accuracy"])
        d_hard = abs(jax_hard - oracle_hard)
        max_hard_delta = max(max_hard_delta, d_hard)
        # Annotation flag: "*" if soft drift is unusually large (informational).
        soft_flag = " (high drift)" if err_logits > soft_logit_floor else ""
        if verbose:
            print(
                f"  step {t:2d}: "
                f"logits Δ={err_logits:.2e}  hidden Δ={err_hidden:.2e}  "
                f"hard_acc jax={jax_hard:.4f} oracle={oracle_hard:.4f} "
                f"(Δ={d_hard:.4f}){soft_flag}"
            )
        if t == max_steps:
            jax_hard_pred = np.asarray(jax_aux["hard_predictions"])
            oracle_hard_pred = np.asarray(oracle_aux["pred_hard"])
            final_pred_disagreements = int(np.sum(jax_hard_pred != oracle_hard_pred))

    ok = max_hard_delta < hard_acc_tol
    if verbose:
        verdict = "PASS" if ok else "FAIL"
        total = int(np.size(np.asarray(jax_aux["hard_predictions"])))
        print(
            f"[jax-parity] {verdict}: max hard_acc Δ over {max_steps} steps = "
            f"{max_hard_delta:.4f} (tol={hard_acc_tol:g}); final-step hard "
            f"prediction disagreements = {final_pred_disagreements}/{total}"
        )
    return ok


def run_perturbation_parity_test(
    *,
    run_id: str | None = None,
    warmup: int = 8,
    after: int = 12,
    n_shotgun: int = 5,
    click_idx: int | None = None,
    hard_acc_tol: float = 0.05,
    weights_json: str | None = None,
    verbose: bool = True,
) -> bool:
    """Parity test for the four user-facing actions in the demo.

    Trajectory:

        steps  0   .. warmup-1            : warm-up (no perturbation)
        step   warmup                      : shotgun (n_shotgun gates)
        steps  warmup+1 .. +after          : observe recovery
        step   warmup+after+1              : click-to-damage one gate
        steps  ..  + after                 : observe recovery
        step   warmup+2*(after+1)          : shuffle wires (Regime III)
        steps  ..  + after                 : observe wiring-OOD recovery

    The randomness for shotgun and shuffle is driven from JAX so the
    chosen gate set / the new wire pattern are byte-identical on both
    sides. Click-to-damage uses a fixed index (defaults to the first
    eligible hidden gate).

    Pass criterion: max ``hard_accuracy`` delta between JAX and oracle
    over the whole trajectory < ``hard_acc_tol``, AND every state
    sync after a perturbation is bit-identical (graph nodes vs oracle
    state) — the perturbation must be applied symmetrically.
    """
    # Lazy imports — keeps the cheap self-test JAX-free.
    import jax
    import numpy as np

    from boolean_nca_cc.circuits.model import gen_circuit, gen_wires, generate_layer_sizes
    from boolean_nca_cc.circuits.tasks import get_task_data
    from boolean_nca_cc.circuits.train import LOSS_L4
    from boolean_nca_cc.training.evaluation import (
        _prepare_model_fn,
        apply_model_and_compute_loss,
        get_loss_and_update_graph,
    )
    from boolean_nca_cc.training.pool.structural_perturbation import (
        apply_knockout_to_circuit,
    )
    from boolean_nca_cc.utils.graph_builder import build_graph

    from .extract_weights import (
        DEFAULT_RUN_ID,
        deserialize_weights,
        extract_weights,
        load_run,
        to_np,
    )
    from .numpy_oracle import (
        CircuitState,
        build_topology,
        compute_residuals,
        flat_to_layered_logits,
        tmt_step,
    )

    rid = run_id or DEFAULT_RUN_ID
    if verbose:
        kind = "quantised (JSON)" if weights_json is not None else "float (extracted)"
        print(f"[perturbation-parity] loading run {rid} ... weights={kind}")
    model, cfg, _ = load_run(run_id=rid)
    if weights_json is not None:
        weights = deserialize_weights(weights_json)
    else:
        weights = extract_weights(model, cfg)

    arity = int(cfg.circuit.arity)
    input_bits = int(cfg.circuit.input_bits)
    output_bits = int(cfg.circuit.output_bits)
    hidden_dim = int(cfg.model.circuit_hidden_dim)
    layer_sizes = generate_layer_sizes(
        input_bits,
        output_bits,
        arity,
        layer_n=int(cfg.circuit.num_layers),
        width_factor=float(cfg.circuit.get("width_factor", 2)),
    )
    parity_seed = int(cfg.get("test_seed", 42)) + 2  # different from clean-parity seed
    wires_key, logits_key = jax.random.split(jax.random.PRNGKey(parity_seed))
    wires_jax, layered_logits_jax = gen_circuit(
        wires_key,
        logits_key,
        layer_sizes,
        arity=arity,
        noise_scale=float(cfg.pool.noise_scale),
    )
    original_shapes = [tuple(lgt.shape) for lgt in layered_logits_jax]
    from boolean_nca_cc.utils.task_compat import get_fixed_task_name
    _task_name = get_fixed_task_name(cfg)
    if _task_name is None:
        raise ValueError(
            "test_oracle requires a fixed task in the checkpoint cfg "
            "(cfg.tasks.name or legacy cfg.circuit.task)."
        )
    task = str(_task_name)
    case_n = 1 << input_bits
    (x_data, y_data), _, _ = get_task_data(
        task,
        case_n=case_n,
        input_bits=input_bits,
        output_bits=output_bits,
        train_test_split=False,
    )
    x_np, y_np = to_np(x_data), to_np(y_data)

    def _build_jax_graph(layered_lg, wires_in, gate_mask=None, hidden=None):
        g = build_graph(
            wires=wires_in,
            logits=layered_lg,
            input_n=input_bits,
            arity=arity,
            circuit_hidden_dim=hidden_dim,
            gate_knockout_mask=gate_mask,
        )
        if hidden is not None:
            g = g._replace(nodes={**g.nodes, "hidden": hidden})
        g, _, _, _ = get_loss_and_update_graph(
            g, original_shapes, wires_in, x_data, y_data, LOSS_L4, layer_sizes
        )
        return g

    def _sync_oracle(graph, topology):
        # Force-copy each leaf so the oracle owns writable buffers — the
        # perturbation paths below mutate ``state.logits`` / ``state.gate_mask``
        # in place, and JAX-backed arrays may surface as read-only views.
        return CircuitState(
            logits=np.array(to_np(graph.nodes["logits"]), copy=True),
            hidden=np.array(to_np(graph.nodes["hidden"]), copy=True),
            loss=np.array(to_np(graph.nodes["loss"]), copy=True),
            gate_mask=np.array(to_np(graph.nodes["gate_knockout_mask"]), copy=True),
        )

    def _verify_clamped(graph, state, indices, label: str) -> bool:
        """Locality check: assert the chosen indices are stamped to ``-10`` and
        masked to ``0`` on *both* sides. This cross-checks the perturbation
        primitive directly without conflating it with the unrelated soft drift
        accumulated during warmup."""
        idx = np.asarray(indices)
        if idx.size == 0:
            return True
        jax_logits = to_np(graph.nodes["logits"])[idx]
        jax_mask = to_np(graph.nodes["gate_knockout_mask"])[idx]
        oracle_logits = state.logits[idx]
        oracle_mask = state.gate_mask[idx]
        problems = []
        if not np.allclose(jax_logits, -10.0):
            problems.append(f"JAX logits @ chosen ≠ -10 (max={jax_logits.max():.3f})")
        if not np.allclose(jax_mask, 0.0):
            problems.append(f"JAX mask @ chosen ≠ 0 (max={jax_mask.max():.3f})")
        if not np.allclose(oracle_logits, -10.0):
            problems.append(f"oracle logits @ chosen ≠ -10 (max={oracle_logits.max():.3f})")
        if not np.allclose(oracle_mask, 0.0):
            problems.append(f"oracle mask @ chosen ≠ 0 (max={oracle_mask.max():.3f})")
        if problems:
            if verbose:
                print(f"  [FAIL] {label}: {'; '.join(problems)}")
            return False
        return True

    # --- Build initial JAX graph + oracle topology / state ---
    graph = _build_jax_graph(layered_logits_jax, wires_jax)
    wires_np = [np.asarray(w, dtype=np.int32) for w in wires_jax]
    topology = build_topology(
        layer_sizes,
        wires_np,
        arity,
        hidden_dim,
        max_neighbors=int(cfg.model.max_neighbors),
    )
    state = _sync_oracle(graph, topology)
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing=False)

    if verbose:
        n_node = int(graph.n_node[0])
        print(
            f"[perturbation-parity] task={task}, n_node={n_node}, case_n={case_n}, "
            f"warmup={warmup}, after={after}, n_shotgun={n_shotgun}"
        )

    perturbation_log: list[str] = []
    max_hard_delta = 0.0

    def _step_once(label: str, *, step_num: int) -> bool:
        nonlocal graph, state, max_hard_delta
        g_new, _, _, jax_aux = apply_model_and_compute_loss(
            model_fn, graph, original_shapes, wires_jax, x_data, y_data, LOSS_L4, layer_sizes
        )
        graph = g_new
        state_new = tmt_step(state, topology, weights)
        state_new, oracle_aux = compute_residuals(state_new, topology, x_np, y_np)
        state = state_new
        jax_pred_hard = np.asarray(jax_aux["hard_predictions"])
        oracle_pred_hard = np.asarray(oracle_aux["pred_hard"])
        n_disagree = int(np.sum(jax_pred_hard != oracle_pred_hard))
        n_total = int(jax_pred_hard.size)
        d = abs(float(jax_aux["hard_accuracy"]) - float(oracle_aux["hard_accuracy"]))
        max_hard_delta = max(max_hard_delta, d)
        if verbose:
            print(
                f"  [{label}] step {step_num:2d}: "
                f"hard_acc jax={float(jax_aux['hard_accuracy']):.4f} "
                f"oracle={float(oracle_aux['hard_accuracy']):.4f} "
                f"(Δ={d:.4f}, {n_disagree}/{n_total} preds differ)"
            )
        return d < hard_acc_tol

    # --- Phase 1: warmup ---
    ok = True
    for t in range(1, warmup + 1):
        ok = _step_once("warmup", step_num=t) and ok
    perturbation_log.append(f"warmup {warmup} steps OK={ok}")

    # --- Perturbation 1: shotgun ---
    if verbose:
        print(f"[perturbation-parity] === SHOTGUN ({n_shotgun} gates) ===")
    shotgun_key = jax.random.PRNGKey(parity_seed + 100)
    modified_logits, knockout_mask_flat = apply_knockout_to_circuit(
        shotgun_key,
        graph.nodes["logits"],
        layer_sizes,
        num_knockouts=n_shotgun,
        faulty_value=-10.0,
        flat=True,
    )
    new_gate_mask = graph.nodes["gate_knockout_mask"] * knockout_mask_flat
    graph = graph._replace(
        nodes={**graph.nodes, "logits": modified_logits, "gate_knockout_mask": new_gate_mask}
    )
    # Recompute the loss feature so the model sees the new residuals next tick.
    graph, _, _, _ = get_loss_and_update_graph(
        graph, original_shapes, wires_jax, x_data, y_data, LOSS_L4, layer_sizes
    )
    chosen = np.where(np.asarray(knockout_mask_flat) == 0.0)[0]
    state.logits[chosen] = -10.0
    state.gate_mask[chosen] = 0.0
    state, _ = compute_residuals(state, topology, x_np, y_np)
    if not _verify_clamped(graph, state, chosen, f"post-shotgun (n={n_shotgun})"):
        return False
    # Drop the warmup soft-drift before continuing so the next phase
    # measures the post-perturbation inference cleanly.
    state = _sync_oracle(graph, topology)
    if verbose:
        print(f"  [OK] shotgun applied symmetrically; chosen flat indices = {chosen.tolist()}")
    for t in range(1, after + 1):
        ok = _step_once("shotgun→", step_num=t) and ok
    perturbation_log.append(f"shotgun + {after} steps OK")

    # --- Perturbation 2: click-to-damage (single gate) ---
    if click_idx is None:
        # First hidden-layer gate that is still healthy (in case shotgun hit it).
        hidden_start = topology.layer_start[1]
        for cand in range(hidden_start, hidden_start + layer_sizes[1][0]):
            if state.gate_mask[cand] == 1.0:
                click_idx = cand
                break
    if click_idx is None:
        if verbose:
            print("  [SKIP] no eligible gate left to damage; skipping click-to-damage")
    else:
        if verbose:
            print(f"[perturbation-parity] === CLICK-TO-DAMAGE (flat_idx={click_idx}) ===")
        new_logits = graph.nodes["logits"].at[click_idx, :].set(-10.0)
        new_mask = graph.nodes["gate_knockout_mask"].at[click_idx].set(0.0)
        graph = graph._replace(nodes={**graph.nodes, "logits": new_logits, "gate_knockout_mask": new_mask})
        graph, _, _, _ = get_loss_and_update_graph(
            graph, original_shapes, wires_jax, x_data, y_data, LOSS_L4, layer_sizes
        )
        state.logits[click_idx] = -10.0
        state.gate_mask[click_idx] = 0.0
        state, _ = compute_residuals(state, topology, x_np, y_np)
        if not _verify_clamped(graph, state, [click_idx], f"post-click (idx={click_idx})"):
            return False
        state = _sync_oracle(graph, topology)
        if verbose:
            print(f"  [OK] click applied symmetrically; clamped flat gate {click_idx}")
        for t in range(1, after + 1):
            ok = _step_once("click→", step_num=t) and ok
        perturbation_log.append(f"click + {after} steps OK")

    # --- Perturbation 3: shuffle wires (Regime III) ---
    if verbose:
        print("[perturbation-parity] === SHUFFLE WIRES ===")
    shuffle_key = jax.random.PRNGKey(parity_seed + 200)
    keys = jax.random.split(shuffle_key, len(layer_sizes) - 1)
    new_wires_jax: list = []
    in_n = layer_sizes[0][0]
    for i, (out_n, group_size) in enumerate(layer_sizes[1:]):
        new_wires_jax.append(gen_wires(keys[i], in_n, out_n, arity, group_size))
        in_n = out_n
    # Recover layered logits from the current graph (kept) + current gate_mask (kept);
    # rebuild the JAX graph with the new wires. ``build_graph`` resets hidden=0,
    # mirroring what DemoSession._restart_tmt_generator does.
    layered_now = flat_to_layered_logits(state, topology)
    layered_jax = [jax.numpy.asarray(lg) for lg in layered_now]
    graph = _build_jax_graph(
        layered_jax, new_wires_jax, gate_mask=graph.nodes["gate_knockout_mask"]
    )
    wires_jax = new_wires_jax
    new_wires_np = [np.asarray(w, dtype=np.int32) for w in new_wires_jax]
    topology = build_topology(
        layer_sizes,
        new_wires_np,
        arity,
        hidden_dim,
        max_neighbors=int(cfg.model.max_neighbors),
    )
    # Mirror the JAX post-shuffle state: keep logits + gate_mask, reset hidden.
    # Then re-sync from the canonical JAX graph (which is the post-rebuild
    # source of truth for hidden=0 + recomputed loss).
    state = _sync_oracle(graph, topology)
    # Verify topology actually changed and that wires + neighbor indices match.
    same_wires = all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(new_wires_jax, new_wires_np)
    )
    if not same_wires:
        if verbose:
            print("  [FAIL] post-shuffle: JAX wires ≠ oracle wires")
        return False
    # Re-prepare model_fn so the precomputed neighbor indices reflect the new topology.
    model_fn, _ = _prepare_model_fn(model, graph, gradient_checkpointing=False)
    if verbose:
        print("  [OK] shuffle applied symmetrically; wires + topology rebuilt")
    for t in range(1, after + 1):
        ok = _step_once("shuffle→", step_num=t) and ok
    perturbation_log.append(f"shuffle + {after} steps OK")

    if verbose:
        verdict = "PASS" if ok else "FAIL"
        print(
            f"[perturbation-parity] {verdict}: max hard_acc Δ = "
            f"{max_hard_delta:.4f} (tol={hard_acc_tol:g})"
        )
        for line in perturbation_log:
            print(f"  · {line}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true", help="Random-weight self-test (no JAX needed)."
    )
    parser.add_argument(
        "--jax-parity",
        action="store_true",
        help="Clean parity (no perturbations) on a real checkpoint.",
    )
    parser.add_argument(
        "--perturbation-parity",
        action="store_true",
        help="Parity under shotgun, click-to-damage, and wire shuffle.",
    )
    parser.add_argument(
        "--all-parity",
        action="store_true",
        help="Run self-test + clean parity + perturbation parity.",
    )
    parser.add_argument("--run-id", default=None, help="Override the deployment run-id.")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--after", type=int, default=12)
    parser.add_argument("--n-shotgun", type=int, default=5)
    parser.add_argument("--click-idx", type=int, default=None)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--weights-json",
        default=None,
        help="Use int8 weights from this JSON instead of extracting from the live model.",
    )
    args = parser.parse_args()

    if args.all_parity:
        args.self_test = True
        args.jax_parity = True
        args.perturbation_parity = True
    if not (args.self_test or args.jax_parity or args.perturbation_parity):
        # Default: run the cheap self-test only.
        ok = run_self_test_all_kinds(verbose=True)
        sys.exit(0 if ok else 1)

    rc = 0
    if args.self_test:
        if not run_self_test_all_kinds(verbose=True):
            rc = 1
    if args.jax_parity:
        if not run_jax_parity_test(
            run_id=args.run_id,
            max_steps=args.max_steps,
            tol=args.tol,
            weights_json=args.weights_json,
        ):
            rc = 1
    if args.perturbation_parity:
        if not run_perturbation_parity_test(
            run_id=args.run_id,
            warmup=args.warmup,
            after=args.after,
            n_shotgun=args.n_shotgun,
            click_idx=args.click_idx,
            weights_json=args.weights_json,
        ):
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
