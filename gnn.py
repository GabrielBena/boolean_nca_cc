from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.circuits.train import loss_f_l4
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.training import train_gnn

from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.train import TrainState
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc import generate_layer_sizes

import jax
import jax.random
import optax
from flax import nnx
import jraph
import jax.numpy as jp

input_n, output_n = 8, 8
arity = 4
layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=4)
loss_type = "l4"
layer_sizes


key = jax.random.PRNGKey(42)
wires, logitsbp = gen_circuit(key, layer_sizes, arity=arity)

opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)
state = TrainState(params=logitsbp, opt_state=opt.init(logitsbp))


case_n = 1 << input_n
x = jp.arange(case_n)
x, y0 = get_task_data(
    "binary_multiply", case_n, input_bits=input_n, output_bits=output_n
)

hidden_dim = 64
hidden_features = 64
n_message_steps = 5
loss, aux = loss_f_l4(logitsbp, wires, x, y0)
graph = build_graph(
    logitsbp, wires, input_n, arity, hidden_dim=hidden_dim, loss_value=loss
)


gnn = CircuitGNN(
    hidden_dim=hidden_dim,
    message_passing=True,
    node_mlp_features=[hidden_features, hidden_features],
    edge_mlp_features=[hidden_features, hidden_features],
    rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    use_attention=True,
    arity=arity,
)

opt_fn = optax.adamw(1e-3, weight_decay=1e-5)
opt = nnx.Optimizer(gnn, opt_fn)


def loss_fn(gnn: CircuitGNN, graph: jraph.GraphsTuple, wires: jax.Array = wires):
    updated_graph = run_gnn_scan(gnn, graph, n_message_steps)
    updated_logits = extract_logits_from_graph(
        updated_graph, [l.shape for l in logitsbp]
    )
    loss, aux = loss_f_l4(updated_logits, wires, x, y0)
    return loss, updated_graph


gnn_results = train_gnn(
    # CIRCUIT PARAMS
    layer_sizes=layer_sizes,
    x_data=x,
    y_data=y0,
    arity=arity,
    # TRAINING PARAMS
    epochs=2**12,
    n_message_steps=n_message_steps,
    key=42,
    # META-LEARNING PARAMS
    meta_batch_size=16,
    meta_learning=True,
    # LOSS PARAMS
    loss_type="l4",
    # LEARNING RATE SCHEDULER
    lr_scheduler="linear_warmup",
    # MESSAGE STEPS SCHEDULE
    message_steps_schedule=None,
    # INITIAL GNN AND OPTIMIZER
    hidden_dim=hidden_dim,
    init_gnn=gnn,
    # init_optimizer=opt,
    # POOL PARAMS
    use_pool=True,
    pool_size=1024,
    reset_pool_fraction=0.01,
    reset_pool_interval=2**4,
)
