# %%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pickle
import jax.numpy as jp
import jax
from boolean_nca_cc.models import CircuitGNN

from flax import nnx

# %%
# define task
from boolean_nca_cc.circuits.tasks import get_task_data
input_n, output_n = 8, 8
case_n = 1 << input_n
x = jp.arange(case_n)
x, y0 = get_task_data(
    "binary_multiply", case_n, input_bits=input_n, output_bits=output_n
)


# %%
# load trained gnn
with open("gnn_pool.pkl", "rb") as f:
    gnn_results = pickle.load(f) 
hidden_dim = 64
hidden_features = 64
arity=4

gnn = CircuitGNN(
    hidden_dim=hidden_dim,
    message_passing=True,
    node_mlp_features=[hidden_features, hidden_features],
    edge_mlp_features=[hidden_features, hidden_features],
    rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    use_attention=False,
    arity=arity,
)
nnx.update(gnn, gnn_results["model"])



# %%
# create circuit
from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
from boolean_nca_cc.circuits.train import loss_f_l4

arity = 4
hidden_dim = 64 # for mlps of hypernetwork
layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=4)
key = jax.random.PRNGKey(42)
wires, logits = gen_circuit(key, layer_sizes, arity=arity)
loss, aux = loss_f_l4(logits, wires, x, y0)

graph = build_graph(
    logits, wires, input_n, arity, hidden_dim=hidden_dim, loss_value=loss
)
## run an inner loop
n_message_steps = 10

def analyse_step(graph, prev_logits, input, output):
    logits = extract_logits_from_graph(
        graph, [l.shape for l in prev_logits]
        )
    loss, aux = loss_f_l4(logits, wires, input, output)
    
    hard_loss = aux["hard_loss"]
    print("Loss is ", loss)
    print("Hard Loss is ", aux["hard_loss"])

    return logits, hard_loss



import numpy as np
hard_losses = []
for step in range(n_message_steps):
    
    print("****Message step*****", step)
    
    current_step_losses = []
    
    for example_input, example_output in zip(x,y0):
    
        logits, loss = analyse_step(graph, logits, example_input, example_output)
        current_step_losses.append(loss)
        break
    hard_losses.append(np.mean(current_step_losses))
    print("Average loss at step " + str(step) + ":", hard_losses[-1])
    graph = gnn(graph)
    
# %%
import matplotlib.pyplot as plt
 
plt.plot(range(n_message_steps),)
    
    
    
    
    


