import types

import jax
import jax.numpy as jp


def test_run_seu_periodic_evaluation_sequential_mode_calls_batched_with_schedule(monkeypatch):
    # Import target function and builder
    from boolean_nca_cc.training.train_loop import run_seu_periodic_evaluation, build_sequential_seu_schedule

    # Minimal fake model: identity over graph
    class FakeModel:
        def __call__(self, g, **kwargs):
            return g

    model = FakeModel()

    # Create trivial base circuit with 1 layer of gates (skip input layer)
    # Wires/logits format matches evaluator expectations: list per layer
    batch_size = 3
    group_n, group_size, arity = 1, 2, 2
    logit_dim = 2 ** arity
    # base_wires is a pytree list; contents not used by this test beyond shape propagation
    base_wires = [jp.zeros((group_n, group_size), dtype=jp.int32) for _ in range(2)]  # input + one gate layer
    base_logits = [jp.zeros((group_n, group_size, logit_dim), dtype=jp.float32)]  # only gate layers carry logits

    # Layer sizes: [(inputs, group_size), (gates, group_size)]
    layer_sizes = [(group_n * group_size, group_size), (group_n * group_size, group_size)]

    # Build a simple schedule from ordered indices and R
    ordered = [0, 1]
    schedule = build_sequential_seu_schedule(
        ordered_indices=ordered,
        layer_sizes=layer_sizes,
        flips_per_gate=1,
        arity=arity,
        recovery_steps=3,
    )

    # Capture arguments passed into evaluate_model_stepwise_batched via chunk wrapper
    captured = {}

    def fake_eval_fn(**kwargs):
        # Store a subset of arguments for assertions
        captured["batch_logits"] = kwargs["logits"]
        captured["seu_schedule"] = kwargs.get("seu_schedule")
        # Return minimal stepwise metrics structure expected by caller
        return {
            "step": [0, 1],
            "soft_loss": [0.0, 0.0],
            "hard_loss": [0.0, 0.0],
            "soft_accuracy": [1.0, 1.0],
            "hard_accuracy": [1.0, 1.0],
        }

    def fake_evaluate_circuits_in_chunks(eval_fn, wires, logits, target_chunk_size, **eval_kwargs):
        # Directly call eval_fn once with provided kwargs
        return fake_eval_fn(**{"wires": wires, "logits": logits, **eval_kwargs})

    # Monkeypatch the chunked evaluator used inside the function under test
    import boolean_nca_cc.training.evaluation as evaluation_mod
    monkeypatch.setattr(evaluation_mod, "evaluate_circuits_in_chunks", fake_evaluate_circuits_in_chunks)

    # Configure sequential mode
    seu_config = {
        "sequential": True,
        "flips_per_gate": 1,
        "arity": arity,
        "recovery_steps": 3,
        "log_stepwise": False,
        "greedy_ordered_indices": ordered,
    }

    # Invoke function under test
    res = run_seu_periodic_evaluation(
        model=model,
        seu_vocabulary=None,
        base_wires=base_wires,
        base_logits=base_logits,
        seu_config=seu_config,
        periodic_eval_test_seed=0,
        x_data=jp.zeros((1,)),
        y_data=jp.zeros((1,)),
        input_n=1,
        arity=arity,
        circuit_hidden_dim=8,
        n_message_steps=5,
        loss_type="l4",
        epoch=0,
        wandb_run=None,
        eval_batch_size=batch_size,
        layer_sizes=layer_sizes,
        layer_neighbors=False,
        greedy_ordered_indices=ordered,
        log_stepwise=False,
    )

    # Assertions: schedule passed through and logits are unmodified base logits replicated
    assert "seu_schedule" in captured and captured["seu_schedule"] is not None
    # Replicated logits must equal base logits repeated along batch dim
    replicated = [jp.repeat(l[None, ...], batch_size, axis=0) for l in base_logits]
    for cap, exp in zip(captured["batch_logits"], replicated):
        assert cap.shape == exp.shape
        assert jp.allclose(cap, exp)

    # Validate return structure carries eval_seu_in keys
    assert "final_metrics_in" in res and "eval_seu_in/final_accuracy" in res["final_metrics_in"]


