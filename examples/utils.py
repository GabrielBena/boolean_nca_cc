from boolean_nca_cc.circuits.train import loss_f_l4
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
import jraph
from boolean_nca_cc.utils import build_graph, extract_logits_from_graph
import jax

def loss_fn(gnn: CircuitGNN, graph: jraph.GraphsTuple, n_message_steps,  wires: jax.Array, logitsbp, x, y0):
    updated_graph = run_gnn_scan(gnn, graph, n_message_steps)
    updated_logits = extract_logits_from_graph(
        updated_graph, [l.shape for l in logitsbp]
    )
    loss, aux = loss_f_l4(updated_logits, wires, x, y0)
    #aux = dict(
    #    act=act, err_mask=err_mask, hard_loss=hard_loss, hard_act=hard_act
    #)
    return loss, updated_graph, aux