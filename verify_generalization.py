import jax
import jax.numpy as jp
import numpy as np
import logging
from boolean_nca_cc.training.checkpointing import load_config_from_wandb, load_model_from_config_and_checkpoint
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise_batched
from boolean_nca_cc.circuits.model import gen_circuit, make_nops
from boolean_nca_cc.circuits.data_split import split_input_combinations
from boolean_nca_cc.circuits.tasks import get_task_data

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def verify_generalization():
    # 1. Load Model
    # Try to find a local checkpoint first
    print("Loading latest model...")
    try:
        # Use a dummy run ID if needed, or let it find the latest local one
        # Assuming typical project structure where checkpoints are in 'checkpoints/' or 'saves/'
        # We'll try to use the load function's default behavior to find the latest
        config, checkpoint_path, run_id = load_config_from_wandb(
            project="boolean-nca-cc",
            entity="marcello-barylli-growai",
            filename="latest_checkpoint",
            run_from_last=1
        )
        
        print(f"Loaded config from run: {run_id}")
        
        model, _ = load_model_from_config_and_checkpoint(
            config=config,
            checkpoint_path=checkpoint_path,
            run_id=run_id
        )
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Data (Binary Multiply Task 12-bit is standard in config)
    # Use config values
    input_bits = config.circuit.input_bits
    output_bits = config.circuit.output_bits
    task_name = config.circuit.task
    
    print(f"Setting up task: {task_name} ({input_bits}->{output_bits})")
    
    # Generate data
    n_cases = 1 << input_bits
    # Limit cases if too large for memory (though 12 bits = 4096 is fine)
    x_data, y_data = get_task_data(task_name, n_cases, input_bits=input_bits, output_bits=output_bits)
    
    # Split train/test (80/20)
    input_train_fraction = 0.8
    seed = 42
    
    x_train, y_train, x_test, y_test = split_input_combinations(
        x_data=x_data,
        y_data=y_data,
        train_fraction=input_train_fraction,
        seed=seed,
        shuffle=True,
    )
    
    print(f"Data split: Train={x_train.shape[0]}, Test={x_test.shape[0]}")
    
    # 3. Setup Circuit
    layer_sizes = config.circuit.layer_sizes
    # If layer_sizes is a ListConfig (OmegaConf), convert to list of tuples
    if hasattr(layer_sizes, '__iter__'):
        layer_sizes = [(item[0], item[1]) for item in layer_sizes]
    
    arity = config.circuit.arity
    wiring_key = jax.random.PRNGKey(42) # Use fixed seed for consistency
    
    # Generate wires and INITIAL logits (NOPs)
    # Crucial: We start from NOPs to see if it can solve the task from scratch
    base_wires, base_logits = gen_circuit(wiring_key, layer_sizes, arity=arity, init_logits_fn=make_nops)
    
    # Batchify for evaluation
    # We'll evaluate a batch of identical circuits on the test set
    batch_size = 16 
    batch_wires = [jp.repeat(w[None, ...], batch_size, axis=0) for w in base_wires]
    batch_logits = [jp.repeat(l[None, ...], batch_size, axis=0) for l in base_logits]
    
    # 4. Run Evaluation - Normal (With Loss Feedback)
    print("\n--- Running Normal Evaluation (Technician Mode) ---")
    results_normal = evaluate_model_stepwise_batched(
        model=model,
        batch_wires=batch_wires,
        batch_logits=batch_logits,
        x_data=x_test,
        y_data=y_test,
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=config.circuit.circuit_hidden_dim,
        n_message_steps=config.training.n_message_steps, # Use training steps
        loss_type=config.training.loss_type,
        layer_sizes=layer_sizes,
        blind_mode=False
    )
    
    final_acc_normal = results_normal["hard_accuracy"][-1]
    print(f"Final Accuracy (Normal): {final_acc_normal:.4f}")
    
    # 5. Run Evaluation - Blind (No Loss Feedback)
    print("\n--- Running Blind Evaluation (Architect Mode) ---")
    results_blind = evaluate_model_stepwise_batched(
        model=model,
        batch_wires=batch_wires,
        batch_logits=batch_logits,
        x_data=x_test, # Still evaluating on test data
        y_data=y_test, 
        input_n=input_bits,
        arity=arity,
        circuit_hidden_dim=config.circuit.circuit_hidden_dim,
        n_message_steps=config.training.n_message_steps,
        loss_type=config.training.loss_type,
        layer_sizes=layer_sizes,
        blind_mode=True # <--- THE CRITICAL SWITCH
    )
    
    final_acc_blind = results_blind["hard_accuracy"][-1]
    print(f"Final Accuracy (Blind): {final_acc_blind:.4f}")
    
    # 6. Conclusion
    print("\n--- Conclusion ---")
    if final_acc_blind < final_acc_normal - 0.1: # Significant drop
        print("RESULT: The Technician Hypothesis is supported.")
        print("The model heavily relies on test-time loss feedback to configure the circuit.")
        print("It essentially runs an optimization algorithm on the test set.")
    else:
        print("RESULT: The Architect Hypothesis is supported.")
        print("The model configures a general-purpose circuit without needing specific feedback on the test inputs.")
        print("This suggests it has learned the structural logic of the task.")

if __name__ == "__main__":
    verify_generalization()

