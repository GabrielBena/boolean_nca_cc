# Randomly wired boolean circuits demo.
# author: Alexander Mordvintsev (moralex@google.com)
# written during The CapoCaccia Workshops toward Neuromorphic Intelligence (https://capocaccia.cc/)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from flax import nnx



import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jp
import optax
import PIL.Image, PIL.ImageDraw
import IPython
from imgui_bundle import (
    implot,
    imgui,
    immapp,
    immvision,
    imgui_ctx,
    hello_imgui,
)

from boolean_nca_cc.utils import build_graph, extract_logits_from_graph


from boolean_nca_cc import generate_layer_sizes
from boolean_nca_cc.circuits.tasks import get_task_data
from boolean_nca_cc.circuits.model import gen_circuit
from boolean_nca_cc.circuits.tasks import TASKS
from examples.utils import loss_fn
from boolean_nca_cc.models import CircuitGNN, run_gnn_scan
from boolean_nca_cc.circuits.train import loss_f_l4
from boolean_nca_cc.training.evaluation import evaluate_model_stepwise

import pickle

input_n, output_n = 8, 8
arity = 4
layer_sizes = generate_layer_sizes(input_n, output_n, arity, layer_n=4)
loss_type = "l4"
case_n = 1 << input_n
x = jp.arange(case_n)
x, y0 = get_task_data(
    "binary_multiply", case_n, input_bits=input_n, output_bits=output_n
)


key = jax.random.PRNGKey(42)
wires, logitsbp = gen_circuit(key, layer_sizes, arity=arity)




################## circuit gate and wire use analysis ##################


def calc_lut_input_use(logits):
    # (..., lut) -> (..., input_use_mask)
    luts = jp.sign(logits) * 0.5 + 0.5
    arity = luts.shape[-1].bit_length() - 1
    luts = luts.reshape(luts.shape[:-1] + (2,) * arity)
    axes_to_flatten = -1 - np.arange(arity - 1)
    input_use = []
    for i in range(1, arity + 1):
        m = luts.take(0, -i) != luts.take(1, -i)
        m = m.any(axes_to_flatten)
        input_use.append(m)
    return jp.stack(input_use)


def propatate_gate_use(input_n, wires, logits, output_use):
    output_use = output_use.reshape(logits.shape[:2])
    gate_input_use = calc_lut_input_use(logits) * output_use
    wire_use_mask = gate_input_use.any(-1)
    used_wires = wires[wire_use_mask]
    prev_gate_use = np.zeros(input_n, np.bool_)
    prev_gate_use[used_wires] = True
    return prev_gate_use, wire_use_mask


def calc_gate_use_masks(input_n, wires, logits):
    layer_sizes = [input_n] + [np.prod(l.shape[:2]) for l in logits]
    gate_use_mask = np.ones(layer_sizes[-1], np.bool_)
    gate_masks = [gate_use_mask]
    wire_masks = []
    for i in range(len(logits))[::-1]:
        gate_use_mask, wire_use_mask = propatate_gate_use(
            layer_sizes[i], wires[i], logits[i], gate_use_mask
        )
        wire_masks.append(wire_use_mask)
        gate_masks.append(gate_use_mask)
    return gate_masks[::-1], wire_masks[::-1]


######################## interactive demo ##############################


def zoom(a, k=2):
    return np.repeat(np.repeat(a, k, 1), k, 0)


def unpack(x, bit_n=8):
    return (x[..., None] >> np.r_[:bit_n]) & 1


def is_point_in_box(p0, p1, p):
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)


max_trainstep_n = 1000


class Demo:
    def __init__(self):
        self.logits0 = logitsbp
        print("param_n:", sum(l.size for l in self.logits0))
        self.logits = self.logits0
        hidden_dim = 64
        hidden_features = 64
        n_message_steps = 5
        loss, aux= loss_f_l4(logitsbp, wires, x, y0)
        
        self.gnn = CircuitGNN(
            hidden_dim=hidden_dim,
            message_passing=True,
            node_mlp_features=[hidden_features, hidden_features],
            edge_mlp_features=[hidden_features, hidden_features],
            rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
            use_attention=True,
            arity=arity,
)
        
        self.step_metrics = evaluate_model_stepwise(
                self.gnn,
                wires,
                logitsbp,
                x,
                y0,
                input_n,
                n_message_steps=100,
                arity=arity,
                hidden_dim=hidden_dim,
                loss_type="l4",
            )
        
        self.wires_key = jax.random.PRNGKey(42)
        self.local_noise = 0.0
        self.shuffle_wires()
        self.reset_gate_mask()

        self.input_x = x
        # Create a proper image format for display - convert to 3-channel uint8
        inp_img = self.input_x.T
        inp_img = np.dstack([inp_img] * 3)  # Convert to 3-channel
        inp_img = zoom(inp_img, 4)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)  # Convert to uint8
        self.active_case_i = 123

        self.tasks = TASKS
        self.task_names = list(self.tasks)
        self.task_idx = self.task_names.index("binary_multiply")
        self.task_text = "All you need are ones  and zeros  and backpropagation"
        self.noise_p = 0
        self.sample_noise()
        self.update_task()

        self.wd_log10 = -1
        self.trainstep_i = 0
        self.is_training = False

    def reset_gate_mask(self):
        self.gate_mask = [np.ones(gate_n) for gate_n, _ in layer_sizes]
        self.wire_masks = [np.ones_like(w, np.bool_) for w in self.wires]

    def mask_unused_gates(self):
        gate_masks, self.wire_masks = calc_gate_use_masks(
            input_n, self.wires, self.logits
        )
        for i in range(len(gate_masks)):
            self.gate_mask[i] = np.array(self.gate_mask[i] * gate_masks[i])

    def shuffle_wires(self):

        self.wires, _ = gen_circuit(key, layer_sizes, arity)

    def sample_noise(self):
        self.noise = np.random.rand(case_n, input_n)

    def update_task(self):
        task_name = self.task_names[self.task_idx]
        if task_name == "text":
            im = PIL.Image.new("L", (case_n, input_n))
            draw = PIL.ImageDraw.Draw(im)
            draw.text((2, -2), self.task_text, fill=255)
            self.y0 = jp.float32(np.array(im) > 100).T
        elif task_name == "noise":
            self.y0 = jp.float32(self.noise < self.noise_p)
        else:
            _, self.y0 = get_task_data(task_name, case_n)

    def draw_gate_lut(self, x, y, logit):
        x0, y0 = x - 20, y - 20 - 36
        dl = imgui.get_window_draw_list()
        lut = jax.nn.sigmoid(logit).reshape(-1, 4)
        col = np.uint32(lut * 255)
        col = (col << 16) | (col << 8) | col | 0xFF000000
        for (i, j), c in np.ndenumerate(col):
            x, y = x0 + j * 10, y0 + i * 10
            dl.add_rect_filled((x, y), (x + 10, y + 10), c)

    def draw_circuit(self, pad=4, d=24, H=600):
        io = imgui.get_io()
        W = imgui.get_content_region_avail().x - pad * 2
        imgui.invisible_button("circuit", (W, H))
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        h = (H - d) / (len(layer_sizes) - 1)
        prev_gate_x = None
        prev_y = 0
        prev_act = None
        case = self.active_case_i
        hover_gate = None
        for li, (gate_n, group_size) in enumerate(layer_sizes):
            group_n = gate_n // group_size
            span_x = W / group_n
            group_w = min(d * group_size, span_x - 6)
            gate_w = group_w / group_size
            group_x = base_x + (np.arange(group_n)[:, None] + 0.5) * span_x
            gate_ofs = (np.arange(group_size) - group_size / 2 + 0.5) * gate_w
            gate_x = (group_x + gate_ofs).ravel()
            y = base_y + li * h + d / 2
            act = np.array(self.act[li][case])
            
            # TODO: need to get actual act
            #act = np.zeros((8,))
            for i, x in enumerate(gate_x):
                a = int(act[i] * 0xA0)
                col = 0xFF202020 + (a << 8)
                p0, p1 = (x - gate_w / 2, y - d / 2), (x + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4)
                if is_point_in_box(p0, p1, io.mouse_pos):
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0)
                    if li > 0:
                        hover_gate = (
                            x,
                            y,
                            self.logits[li - 1][i // group_size, i % group_size],
                        )
                    if io.mouse_clicked[0]:
                        if li > 0:
                            self.gate_mask[li][i] = 1.0 - self.gate_mask[li][i]
                        else:
                            self.active_case_i = self.active_case_i ^ (1 << i)
                if self.gate_mask[li][i] == 0.0:
                    dl.add_rect_filled(p0, p1, 0xA00000FF, 4)

            for x in group_x[:, 0]:
                dl.add_rect(
                    (x - group_w / 2, y - d / 2),
                    (x + group_w / 2, y + d / 2),
                    0x80FFFFFF,
                    4,
                )

            if li > 0:
                wires = self.wires[li - 1].T
                masks = self.wire_masks[li - 1].T
                src_x = prev_gate_x[wires]
                dst_x = (
                    group_x + (np.arange(arity) + 0.5) / arity * group_w - group_w / 2
                )
                my = (prev_y + y) / 2
                for x0, x1, si, m in zip(
                    src_x.ravel(), dst_x.ravel(), wires.ravel(), masks.ravel()
                ):
                    if not m:
                        continue
                    a = int(prev_act[si] * 0x60)
                    col = 0xFF404040 + (a << 8)
                    dl.add_bezier_cubic(
                        (x0, prev_y + d / 2),
                        (x0, my),
                        (x1, my),
                        (x1, y - d / 2),
                        col,
                        1.0,
                    )
            if hover_gate is not None:
                self.draw_gate_lut(*hover_gate)

            prev_gate_x = gate_x
            prev_act = act
            prev_y = y

    def draw_lut(self, name, img):
        view_w = imgui.get_content_region_avail().x
        img_h, img_w = img.shape[:2]
        print(f"Image shape: {img.shape}")
        view_w = imgui.get_content_region_avail().x
        print(f"Available content region width: {view_w}")
        mx, _ = immvision.image_display_resizable(
            name, img, (view_w, 0), resizable=False, refresh_image=True
        )
        
        #except RuntimeError:
        #    print("check")
        if mx > 0.0 and mx < img_w:
            self.active_case_i = int(mx / img_w * case_n)
        x0, y0 = imgui.get_item_rect_min()
        x1, y1 = imgui.get_item_rect_max()
        x = x0 + (x1 - x0) * (self.active_case_i + 0.5) / case_n
        imgui.get_window_draw_list().add_line((x, y0), (x, y1), 0x8000FF00, 2.0)

    def update_circuit(self):
        hidden_dim = 64
        hidden_features = 64
        n_message_steps = 5
        loss, aux= loss_f_l4(logitsbp, wires, x, y0)
        
        
        self.act = aux["act"]
        oimg = self.act[-1].T
        oimg = np.dstack([oimg] * 3)
        m = aux["err_mask"].T[..., None] * 0.5

        oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])
        oimg = zoom(oimg, 4)

        self.outputs_img = np.uint8(oimg.clip(0, 1) * 255)
        graph = build_graph(
        logitsbp, wires, input_n, arity, hidden_dim=hidden_dim, loss_value=0)


        #opt = optax.adamw(1, 0.8, 0.8, weight_decay=1e-1)

        #(loss, graph, aux), grad = nnx.value_and_grad(loss_fn, has_aux=True)(gnn, graph, logitsbp=logitsbp, wires=wires, n_message_steps=n_message_steps,x=x,y0=y0)

        with open("gnn_results.pkl", "rb") as f:
            gnn_results = pickle.load(f)

        if self.is_training:
            self.step_metrics = evaluate_model_stepwise(
                self.gnn,
                wires,
                logitsbp,
                x,
                y0,
                input_n,
                n_message_steps=100,
                arity=arity,
                hidden_dim=hidden_dim,
                loss_type="l4",
            )
        # here we take just the last step of message passing
        last_step_metrics = {}
        for key, all_steps in self.step_metrics.items():
            
            last_step_metrics[key] = all_steps[-1]
        """    
        aux = [
            {"accuracy": acc, "hard_accuracy": hard_acc, "hard_loss": hard_loss}
            for acc, hard_acc, hard_loss in zip(
                step_metrics["soft_accuracy"],
                step_metrics["hard_accuracy"],
                step_metrics["hard_loss"],
            )
        ]
        """
        aux = {"act": last_step_metrics["acts"]}

        self.act = aux["act"]
        

        

            
    def gui(self):
        self.update_circuit()
        runner_params = hello_imgui.get_runner_params()
        runner_params.fps_idling.enable_idling = True
        io = imgui.get_io()

        imgui.begin_child("main", (-200, 0))

        imgui.separator_text("Inputs")
        self.draw_lut("inputs", self.inputs_img)

        H = imgui.get_content_region_avail().y - 100
        self.draw_circuit(H=H)

        imgui.separator_text("Outputs")
        self.draw_lut("outputs", self.outputs_img)

        imgui.end_child()
        imgui.same_line()

        imgui.begin_child("controls")

        if imgui.button("python REPL"):
            IPython.embed()

        _, self.is_training = imgui.checkbox("is_training", self.is_training)
        if imgui.button("reset gates"):
            self.logits = self.logits0
            self.trainstep_i = 0
        if imgui.button("reset gates + opt"):
            self.logits = self.logits0
            self.opt_state = self.get_opt().init(self.logits)
            self.trainstep_i = 0
        if imgui.button("shuffle wires"):
            self.wires_key, key = jax.random.split(self.wires_key)
            self.shuffle_wires()
            self.trainstep_i = 0
        local_noise_changed, self.local_noise = imgui.slider_float(
            "local noise", self.local_noise, 0.0, 20.0
        )
        if local_noise_changed:
            self.shuffle_wires()

        _, self.wd_log10 = imgui.slider_float("wd_log10", self.wd_log10, -3, 0.0)

        imgui.separator_text("Masks")
        if imgui.button("reset gate mask"):
            self.reset_gate_mask()
        if imgui.button("mask unused gates"):
            self.mask_unused_gates()
        active_gate_n = int(sum(m.sum() for m in self.gate_mask))
        imgui.text(f"active gate n: {active_gate_n}")

        imgui.separator_text("Task")
        task_changed, self.task_idx = imgui.combo(
            "task", self.task_idx, self.task_names
        )
        if task_changed:
            self.update_task()
            self.trainstep_i = 0
        task_name = self.task_names[self.task_idx]
        if task_name == "text":
            text_changed, self.task_text = imgui.input_text("text", self.task_text)
            if text_changed:
                self.update_task()
        if task_name == "noise":
            noise_changed, self.noise_p = imgui.slider_float(
                "p", self.noise_p, 0.0, 1.0
            )
            if noise_changed:
                self.update_task()

        imgui.end_child()


if __name__ == "__main__":
    demo = Demo()

    immvision.use_rgb_color_order()
    immapp.run(
        demo.gui,
        window_title="Graph NCA",
        window_size=(800, 600),
        #window_size_auto=True,
        window_restore_previous_geometry=False,
        fps_idle=10,
        with_implot=True,
    )  # type: ignore
