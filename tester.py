from boolean_nca_cc.training.utils import load_best_model_from_wandb

ch = load_best_model_from_wandb(
    filters={
        "config.model.type": "gnn",
        "config.circuit.arity": 2,
        "config.circuit.num_layers": 3,
        "config.circuit.input_bits": 4,
        "config.circuit.output_bits": 4,
    }
)