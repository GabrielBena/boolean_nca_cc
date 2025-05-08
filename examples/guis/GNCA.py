# Randomly wired boolean circuits demo.
# author: Alexander Mordvintsev (moralex@google.com)
# written during The CapoCaccia Workshops toward Neuromorphic Intelligence (https://capocaccia.cc/)


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
    imgui_knobs,
    imgui,
    immapp,
    imgui_ctx,
    immvision,
    hello_imgui,
)


################## boolear circuit definition ##################

input_n, output_n = 8, 8
case_n = 1 << input_n
arity, layer_width, layer_n = 4, 64, 5
layer_sizes = (
    [(input_n, 1)]
    + [(layer_width, arity)] * (layer_n - 1)
    + [(layer_width // 2, arity // 2), (output_n, 1)]
)


def gen_wires(key, in_n, out_n, arity, group_size, local_noise=None):
    edge_n = out_n * arity // group_size
    if in_n != edge_n or local_noise is None:
        n = max(in_n, edge_n)
        return jax.random.permutation(key, n)[:edge_n].reshape(arity, -1) % in_n
    i = (
        jp.arange(edge_n) + jax.random.normal(key, shape=(edge_n,)) * local_noise
    ).argsort()
    return i.reshape(-1, arity).T


def make_nops(gate_n, arity, group_size, nop_scale=3.0):
    I = jp.arange(1 << arity)
    bits = (I >> I[:arity, None]) & 1
    luts = bits[jp.arange(gate_n) % arity]
    logits = (2.0 * luts - 1.0) * nop_scale
    return logits.reshape(gate_n // group_size, group_size, -1)


@jax.jit
def run_layer(lut, inputs):
    # lut:[group_n, group_size, 1<<arity], [arity, ... , group_n]
    for x in inputs:
        x = x[..., None, None]
        lut = (1.0 - x) * lut[..., ::2] + x * lut[..., 1::2]
    # [..., group_n, group_size, 1]
    return lut.reshape(*lut.shape[:-3] + (-1,))


def run_circuit(logits, wires, gate_mask, x, hard=False):
    x = x * gate_mask[0]
    acts = [x]
    for ws, lgt, mask in zip(wires, logits, gate_mask[1:]):
        luts = jax.nn.sigmoid(lgt)
        if hard:
            luts = jp.round(luts)
        x = run_layer(luts, [x[..., w] for w in ws]) * mask
        acts.append(x)
    return acts


def res2loss(res):
    return jp.square(jp.square(res)).sum()


def loss_f(logits, wires, gate_mask, x, y0):
    run_f = partial(run_circuit, logits, wires, gate_mask, x)
    act = run_f()
    loss = res2loss(act[-1] - y0)
    hard_act = run_f(hard=True)
    hard_loss = res2loss(hard_act[-1] - y0)
    err_mask = hard_act[-1] != y0
    return loss, dict(
        act=act, err_mask=err_mask, hard_loss=hard_loss, hard_act=hard_act
    )


grad_loss_f = jax.jit(jax.value_and_grad(loss_f, has_aux=True))


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
        self.logits0 = []
        for gate_n, group_size in layer_sizes[1:]:
            self.logits0.append(make_nops(gate_n, arity, group_size))
        self.logits = self.logits0
        print("param_n:", sum(l.size for l in self.logits0))
        self.wires_key = jax.random.PRNGKey(42)
        self.local_noise = 0.0
        self.shuffle_wires()
        self.reset_gate_mask()

        x = jp.arange(case_n)
        self.input_x = unpack(x)
        # Create a proper image format for display - convert to 3-channel uint8
        inp_img = self.input_x.T
        inp_img = np.dstack([inp_img] * 3)  # Convert to 3-channel
        inp_img = zoom(inp_img, 4)
        self.inputs_img = np.uint8(inp_img.clip(0, 1) * 255)  # Convert to uint8
        self.active_case_i = 123

        self.tasks = dict(
            copy=x,
            gray=x ^ (x >> 1),
            add4=(x & 0xF) + (x >> 4),
            mul4=(x & 0xF) * (x >> 4),
            popcount=np.bitwise_count(x),
            text=x,
            noise=x,
        )
        self.task_names = list(self.tasks)
        self.task_idx = self.task_names.index("mul4")
        self.task_text = "All you need are ones  and zeros  and backpropagation"
        self.noise_p = 0.5
        self.sample_noise()
        self.update_task()

        self.wd_log10 = -1
        self.opt_state = self.get_opt().init(self.logits)
        self.loss_log = np.zeros(max_trainstep_n, np.float32)
        self.hard_log = np.zeros(max_trainstep_n, np.float32)
        self.trainstep_i = 0
        self.is_training = True

    def get_opt(self):
        return optax.adamw(2.0, 0.8, 0.8, weight_decay=10**self.wd_log10)

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
        in_n = input_n
        self.wires = []
        key = self.wires_key
        for gate_n, group_size in layer_sizes[1:]:
            key, k1 = jax.random.split(key)
            local_noise = self.local_noise if self.local_noise > 0.0 else None
            ws = gen_wires(k1, in_n, gate_n, arity, group_size, local_noise)
            self.wires.append(ws)
            in_n = gate_n

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
            self.y0 = jp.float32(unpack(self.tasks[task_name]))

    def update_circuit(self):
        (loss, aux), grad = grad_loss_f(
            self.logits, self.wires, self.gate_mask, self.input_x, self.y0
        )
        if self.is_training:
            upd, self.opt_state = self.get_opt().update(
                grad, self.opt_state, self.logits
            )
            self.logits = optax.apply_updates(self.logits, upd)

        self.act = aux["act"]
        oimg = self.act[-1].T
        oimg = np.dstack([oimg] * 3)
        m = aux["err_mask"].T[..., None] * 0.5

        oimg = oimg * (1.0 - m) + m * np.float32([1, 0, 0])
        oimg = zoom(oimg, 4)

        self.outputs_img = np.uint8(oimg.clip(0, 1) * 255)

        hard_loss = aux["hard_loss"].item()
        i = self.trainstep_i % len(self.loss_log)
        self.loss_log[i] = loss
        self.hard_log[i] = hard_loss
        if self.is_training:
            self.trainstep_i += 1

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

    def gui(self):
        self.update_circuit()
        runner_params = hello_imgui.get_runner_params()
        runner_params.fps_idling.enable_idling = True
        io = imgui.get_io()

        imgui.begin_child("main", (-200, 0))

        if implot.begin_plot("Train logs", (-1, 200)):
            implot.setup_legend(implot.Location_.north_east.value)
            implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
            implot.plot_line("loss", self.loss_log)
            implot.plot_line("hard_loss", self.hard_log)
            implot.drag_line_x(
                1, self.trainstep_i % len(self.loss_log), (0.8, 0, 0, 0.5)
            )
            implot.end_plot()

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
