import numpy as np
import jax
import jax.numpy as jp
import PIL.Image, PIL.ImageDraw
import IPython # Keep for now, for the REPL button

from imgui_bundle import (
    implot,
    imgui,
    immapp,
    immvision,
    hello_imgui,
)

# Helper functions (copied from random_wires_demo.py)
def zoom(a, k=2):
    return np.repeat(np.repeat(a, k, 1), k, 0)


def unpack(x, bit_n=8):
    return (x[..., None] >> np.r_[:bit_n]) & 1


def is_point_in_box(p0, p1, p):
    (x0, y0), (x1, y1), (x, y) = p0, p1, p
    return (x0 <= x <= x1) and (y0 <= y <= y1)


class CircuitVisualizer:
    def __init__(self, layer_sizes, arity, case_n, input_n,
                 # Callbacks for interactive elements
                 on_toggle_gate_mask=None,
                 on_set_active_case_i=None,
                 on_python_repl=None,
                 on_toggle_training=None,
                 on_reset_gates=None,
                 on_reset_gates_opt=None,
                 on_shuffle_wires=None,
                 on_set_local_noise=None,
                 on_set_wd_log10=None,
                 on_reset_gate_mask_action=None, # Renamed to avoid conflict
                 on_mask_unused_gates=None,
                 on_set_task_idx=None,
                 on_set_task_text=None,
                 on_set_noise_p=None
                 ):
        # Static circuit parameters (or parameters that define structure)
        self.layer_sizes = layer_sizes
        self.arity = arity
        self.case_n = case_n
        self.input_n = input_n # Needed for draw_circuit click action on input layer

        # Data to be visualized (updated from outside)
        self.logits = []  # List of JAX arrays
        self.wires = []   # List of JAX arrays
        self.wire_masks = [] # List of numpy bool arrays
        self.gate_mask = [] # List of numpy float arrays (0.0 or 1.0)
        self.act = []     # List of JAX arrays (activations)
        self.inputs_img = np.zeros((1,1,3), dtype=np.uint8) # Placeholder
        self.outputs_img = np.zeros((1,1,3), dtype=np.uint8) # Placeholder
        self.active_case_i = 0
        self.loss_log = np.zeros(1000, np.float32) # Max_trainstep_n was 1000
        self.hard_log = np.zeros(1000, np.float32)
        self.trainstep_i = 0
        self.is_training = True
        self.wd_log10 = -1.0
        self.local_noise = 0.0
        self.task_idx = 0
        self.task_names = ["dummy_task"]
        self.task_text = "dummy text"
        self.noise_p = 0.5
        
        # Store references to initial data if needed by callbacks, e.g. logits0 for reset
        self.logits0_ref = [] # This would need to be passed if "reset gates" implies original logits

        # Callbacks
        self.on_toggle_gate_mask = on_toggle_gate_mask
        self.on_set_active_case_i = on_set_active_case_i
        self.on_python_repl = on_python_repl
        self.on_toggle_training = on_toggle_training
        self.on_reset_gates = on_reset_gates
        self.on_reset_gates_opt = on_reset_gates_opt
        self.on_shuffle_wires = on_shuffle_wires
        self.on_set_local_noise = on_set_local_noise
        self.on_set_wd_log10 = on_set_wd_log10
        self.on_reset_gate_mask_action = on_reset_gate_mask_action
        self.on_mask_unused_gates = on_mask_unused_gates
        self.on_set_task_idx = on_set_task_idx
        self.on_set_task_text = on_set_task_text
        self.on_set_noise_p = on_set_noise_p

    def update_data(self, logits=None, wires=None, wire_masks=None, gate_mask=None, act=None,
                    inputs_img=None, outputs_img=None, active_case_i=None,
                    loss_log=None, hard_log=None, trainstep_i=None,
                    is_training=None, wd_log10=None, local_noise=None,
                    task_idx=None, task_names=None, task_text=None, noise_p=None,
                    logits0_ref=None):
        if logits is not None: self.logits = logits
        if wires is not None: self.wires = wires
        if wire_masks is not None: self.wire_masks = wire_masks
        if gate_mask is not None: self.gate_mask = gate_mask
        if act is not None: self.act = act
        if inputs_img is not None: self.inputs_img = inputs_img
        if outputs_img is not None: self.outputs_img = outputs_img
        if active_case_i is not None: self.active_case_i = active_case_i
        if loss_log is not None: self.loss_log = loss_log
        if hard_log is not None: self.hard_log = hard_log
        if trainstep_i is not None: self.trainstep_i = trainstep_i
        if is_training is not None: self.is_training = is_training
        if wd_log10 is not None: self.wd_log10 = wd_log10
        if local_noise is not None: self.local_noise = local_noise
        if task_idx is not None: self.task_idx = task_idx
        if task_names is not None: self.task_names = task_names
        if task_text is not None: self.task_text = task_text
        if noise_p is not None: self.noise_p = noise_p
        if logits0_ref is not None: self.logits0_ref = logits0_ref

    def draw_gate_lut(self, x, y, logit):
        x0, y0 = x - 20, y - 20 - 36
        dl = imgui.get_window_draw_list()
        # Ensure logit is a JAX array for sigmoid, then convert to numpy for manipulation
        # If logits are already sigmoided and reshaped, this needs adjustment.
        # Assuming 'logit' here is the raw logit for a single gate.
        lut = jax.nn.sigmoid(jp.array(logit)).reshape(-1, 4) # logit might be (1 << arity)
        col = np.uint32(np.array(lut) * 255) # convert to numpy for bitwise operations
        col = (col << 16) | (col << 8) | col | 0xFF000000
        for (i, j), c in np.ndenumerate(col):
            lx, ly = x0 + j * 10, y0 + i * 10 # Use lx, ly to avoid conflict with outer x,y
            dl.add_rect_filled((lx, ly), (lx + 10, ly + 10), c)

    def draw_circuit(self, pad=4, d=24, H=600):
        io = imgui.get_io()
        W = imgui.get_content_region_avail().x - pad * 2
        imgui.invisible_button("circuit_panel", (W, H)) # Renamed button to avoid conflict with potential parent window
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        h = (H - d) / (len(self.layer_sizes) - 1) if len(self.layer_sizes) > 1 else 0
        prev_gate_x = None
        prev_y = 0
        prev_act = None
        case = self.active_case_i
        hover_gate = None

        if not self.act or not self.layer_sizes: # Not enough data to draw
            return

        for li, (gate_n, group_size) in enumerate(self.layer_sizes):
            group_n = gate_n // group_size
            span_x = W / group_n if group_n > 0 else W
            group_w = min(d * group_size, span_x - 6)
            gate_w = group_w / group_size if group_size > 0 else group_w
            group_x_centers = base_x + (np.arange(group_n) + 0.5) * span_x
            
            # gate_ofs needs group_size > 0
            gate_ofs_single_group = (np.arange(group_size) - group_size / 2 + 0.5) * gate_w if group_size > 0 else np.array([0.0])

            # Ensure group_x_centers is 2D for broadcasting with gate_ofs_single_group
            gate_x = (group_x_centers[:, None] + gate_ofs_single_group).ravel()

            y = base_y + li * h + d / 2
            
            current_act_layer = self.act[li]
            # Ensure current_act_layer is a numpy array and has the expected structure.
            # It should be [case_n_or_1, num_gates_in_layer]
            # We need act[case] which implies act might be [num_layers, case_n, num_gates_total_layer]
            # or act is a list of [case_n, num_gates_in_layer]
            # The original code uses self.act[li][case]
            if case >= len(current_act_layer): # Safety check if active_case_i is out of bounds for this layer's activations
                 act_values_for_case = np.zeros(gate_n) # Default to zeros
            else:
                 act_values_for_case = np.array(current_act_layer[case])


            if len(act_values_for_case) != gate_n: # Safety check
                act_values_for_case = np.zeros(gate_n)


            for i, x_coord in enumerate(gate_x):
                # act_values_for_case should have `gate_n` elements.
                activation_value = act_values_for_case[i] if i < len(act_values_for_case) else 0.0
                
                a = int(activation_value * 0xA0)
                col = 0xFF202020 + (a << 8) # Potential error if a is not correctly calculated
                p0, p1 = (x_coord - gate_w / 2, y - d / 2), (x_coord + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4)

                is_hovered = is_point_in_box(p0, p1, io.mouse_pos)
                if is_hovered:
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0)
                    if li > 0 and i < len(self.logits[li-1].flat): # Check bounds for logits
                        # Logits are [layer_idx-1][group_idx, element_in_group_idx]
                        # i is the flattened index for the current layer li
                        logit_val = self.logits[li - 1].reshape(-1)[i // group_size, i % group_size] # This indexing might be wrong
                        # Original: self.logits[li - 1][i // group_size, i % group_size]
                        # self.logits is a list of arrays, each shaped (group_n, group_size, lut_size)
                        # So we need to access the correct logit based on its group and position in group.
                        
                        # Ensure logits for layer li-1 exist and i is a valid index
                        if self.logits and li > 0 and (li-1) < len(self.logits):
                            current_layer_logits = self.logits[li-1] # Shape (group_n_prev_layer, group_size_prev_layer, lut_size)
                            # This needs to map 'i' (gate index in current layer) to a logit in the *previous* layer's output
                            # This logic is complex as 'i' is for the current layer 'li' of nodes,
                            # but logits are associated with connections/gates *between* layers.
                            # The original `hover_gate` logic used `self.logits[li - 1][i // group_size, i % group_size]`
                            # This implies `i` here should refer to the gate index *in the context of the logits array for that layer*.
                            # The current `i` is an index for the *nodes* in layer `li`.
                            # A gate's output becomes a node in the next layer. So, node `i` in layer `li` corresponds to gate `i` in layer `li-1`.
                            num_groups_in_logit_layer = current_layer_logits.shape[0]
                            num_in_group_logit_layer = current_layer_logits.shape[1]
                            
                            if (i // num_in_group_logit_layer) < num_groups_in_logit_layer and \
                               (i % num_in_group_logit_layer) < num_in_group_logit_layer:
                                logit_val_for_hover = current_layer_logits[i // num_in_group_logit_layer, i % num_in_group_logit_layer]
                                hover_gate = (x_coord, y, logit_val_for_hover)


                if io.mouse_clicked[0] and is_hovered:
                    if li > 0: # Not input layer
                        if self.on_toggle_gate_mask:
                             # Needs layer index (li) and gate index within layer (i)
                            self.on_toggle_gate_mask(li, i)
                    else: # Input layer
                        if self.on_set_active_case_i:
                            # Toggle the i-th bit of active_case_i
                            # self.input_n is the number of input bits
                            if i < self.input_n: # Make sure we are clicking a valid input bit
                                self.on_set_active_case_i(self.active_case_i ^ (1 << i))
                
                # Check gate_mask for layer li, gate i
                if li < len(self.gate_mask) and i < len(self.gate_mask[li]):
                    if self.gate_mask[li][i] == 0.0:
                        dl.add_rect_filled(p0, p1, 0xA00000FF, 4)

            for x_center_group in group_x_centers:
                dl.add_rect(
                    (x_center_group - group_w / 2, y - d / 2),
                    (x_center_group + group_w / 2, y + d / 2),
                    0x80FFFFFF,
                    4,
                )

            if li > 0:
                if not self.wires or (li-1) >= len(self.wires) or \
                   not self.wire_masks or (li-1) >= len(self.wire_masks) or \
                   prev_gate_x is None or prev_act is None:
                    pass # Skip drawing wires if data is missing
                else:
                    current_wires = self.wires[li - 1].T # Wires connecting prev_layer to current_layer (li)
                    current_wire_masks = self.wire_masks[li - 1].T

                    # Ensure prev_gate_x has enough elements for wire indexing
                    # Wires are indices into prev_gate_x
                    valid_wire_indices = current_wires < len(prev_gate_x)
                    src_x = prev_gate_x[current_wires[valid_wire_indices]]
                    
                    # dst_x calculation needs arity
                    # Original: group_x + (np.arange(arity) + 0.5) / arity * group_w - group_w / 2
                    # This means arity wires go into each *group* of the current layer
                    # group_x_centers are the centers of the groups in the current layer
                    # Let's assume dst_x calculation needs to be carefully mapped.
                    # The original code's dst_x implied multiple input wires per group, distributed across group_w
                    # For now, let's simplify or be very careful.
                    # dst_x should be `gate_x` for the current layer, as wires connect to gates.
                    # However, the original code had a specific dst_x structure using arity.

                    # Replicating: dst_x = (group_x_centers[:,None] + (np.arange(self.arity) + 0.5) / self.arity * group_w - group_w / 2)
                    # This creates destinations per group, based on arity.
                    # Need to match src_x.ravel() with dst_x.ravel()
                    
                    # A simpler interpretation: wires connect from a source gate in prev_layer to a destination gate in current_layer.
                    # The `wires` array (self.wires[li-1]) has shape (arity, num_gates_in_current_layer_if_grouped)
                    # Let's use the flattened gate_x of the current layer for destinations.
                    # Each gate in the current layer `li` receives `arity` inputs.
                    # `wires[li-1]` is `(arity, n_groups_curr_layer * n_gates_per_group_curr_layer / arity_factor??)`
                    # This part is tricky without fully re-deriving wire logic.
                    # The original code's `wires.ravel()` and `masks.ravel()` iterated through all potential wire endpoints.
                    
                    # Let's stick to the original loop structure as much as possible
                    # `wires = self.wires[li-1].T` -> shape `(num_dest_groups * arity_per_dest_group / group_size_factor, arity)` ... this is complex
                    # Let `wires` be `self.wires[li-1]`, shape `(arity, num_output_gates_in_layer)`
                    # Original: `wires = self.wires[li - 1].T` makes it `(num_output_gates_in_layer, arity)`
                    # `src_x = prev_gate_x[wires]` this means each row of `wires` contains indices into `prev_gate_x`
                    
                    # `dst_x` in original was complex. Let's try to map wires to current layer's gates.
                    # If `self.wires[li-1]` has shape `(arity, N_curr)`, where N_curr is #gates in current layer.
                    # Then `wires.T` is `(N_curr, arity)`.
                    # `src_x = prev_gate_x[wires.T]` would give sources for each of `arity` inputs for `N_curr` gates.
                    # `dst_x` would be `np.repeat(gate_x, self.arity)` if wires.T is (N_curr, arity)

                    # Let's assume `self.wires[li-1]` is `(arity, n_output_nodes_in_layer_li)`
                    # And `self.wire_masks[li-1]` is `(arity, n_output_nodes_in_layer_li)`
                    # `n_output_nodes_in_layer_li` is `gate_n` of current layer `li`.

                    if current_wires.ndim == 2 and current_wires.shape[0] > 0 and current_wires.shape[1] > 0:
                        num_dest_gates_in_layer_li = current_wires.shape[0] # This should be gate_n for layer li
                        
                        # Iterate over destination gates in the current layer
                        for dest_gate_idx_in_li in range(num_dest_gates_in_layer_li):
                            # For each dest_gate, it has `arity` incoming wires
                            # Original `wires.T` means `wires[dest_gate_idx_in_li]` are the source gate indices
                            source_gate_indices_for_dest = current_wires[dest_gate_idx_in_li] # Shape (arity,)
                            mask_values_for_dest = current_wire_masks[dest_gate_idx_in_li] # Shape (arity,)
                            
                            dest_x_coord = gate_x[dest_gate_idx_in_li] # x-coord of the current destination gate in layer li

                            for input_idx_to_dest_gate in range(self.arity): # Iterate over each of the `arity` inputs
                                if not mask_values_for_dest[input_idx_to_dest_gate]:
                                    continue
                                
                                source_gate_idx_in_prev_layer = source_gate_indices_for_dest[input_idx_to_dest_gate]
                                if source_gate_idx_in_prev_layer >= len(prev_gate_x): # Bounds check
                                    continue

                                src_x_coord = prev_gate_x[source_gate_idx_in_prev_layer]
                                
                                # The original dst_x for bezier curve was more nuanced, distributing inputs visually.
                                # For now, let's draw to the center of the dest_gate_idx_in_li
                                # To replicate the spread:
                                # `dst_x_for_wire = group_x_centers[dest_gate_idx_in_li // group_size] + \
                                #                   (input_idx_to_dest_gate - self.arity / 2 + 0.5) * (gate_w / self.arity_if_wires_are_grouped_like_that)`
                                # This needs careful thought on how group_w and arity map for wire destinations.
                                # The original `dst_x` was:
                                # group_x_for_dest_layer = base_x + (np.arange(group_n) + 0.5) * span_x # group centers of current layer li
                                # dst_x_spread = (group_x_for_dest_layer[:, None] + \
                                #                (np.arange(self.arity) - self.arity/2 + 0.5) * (group_w / self.arity) # This assumes group_w is for one group
                                #                ).ravel() 
                                # This dst_x_spread would need to be indexed correctly.
                                # Let's use the simpler direct connection to dest_x_coord for now. A visual refinement might be needed.
                                final_dst_x_coord = dest_x_coord # Simplified: wire goes to center of dest gate.
                                                             # To make it spread like original:
                                                             # Find group_idx for dest_gate_idx_in_li:
                                current_gate_group_idx = dest_gate_idx_in_li // group_size
                                current_gate_pos_in_group = dest_gate_idx_in_li % group_size
                                
                                # Visual distribution of wire endpoints across the width of the *group* the dest_gate belongs to
                                # This requires `group_w` of the current layer for the specific group
                                # `group_w` was min(d * group_size, span_x - 6)
                                # The arity inputs are spread over the gate, not group. So use gate_w.
                                wire_offset_on_gate = (input_idx_to_dest_gate - self.arity / 2 + 0.5) * (gate_w / self.arity) if self.arity > 0 else 0
                                final_dst_x_coord = dest_x_coord + wire_offset_on_gate


                                activation_of_source_gate = prev_act[source_gate_idx_in_prev_layer] if source_gate_idx_in_prev_layer < len(prev_act) else 0.0
                                a_wire = int(activation_of_source_gate * 0x60)
                                col_wire = 0xFF404040 + (a_wire << 8)
                                my = (prev_y + y) / 2
                                dl.add_bezier_cubic(
                                    (src_x_coord, prev_y + d / 2),
                                    (src_x_coord, my),
                                    (final_dst_x_coord, my),
                                    (final_dst_x_coord, y - d / 2),
                                    col_wire,
                                    1.0,
                                )

            if hover_gate is not None:
                self.draw_gate_lut(*hover_gate)

            prev_gate_x = gate_x
            prev_act = act_values_for_case # This should be the activations of the current layer for the selected case
            prev_y = y

    def draw_lut(self, name, img_data_to_display): # Renamed img to img_data_to_display
        if img_data_to_display is None or img_data_to_display.size == 0:
            imgui.text(f"{name} data not available.")
            return

        view_w = imgui.get_content_region_avail().x
        img_h, img_w = img_data_to_display.shape[:2]
        
        # immvision.image_display_resizable expects a name that is unique for the image texture
        unique_img_name = f"{name}_display_##{id(img_data_to_display)}"

        # Ensure image is contiguous for immvision if it's coming from complex numpy ops
        display_img_cont = np.ascontiguousarray(img_data_to_display)

        mx, _ = immvision.image_display_resizable(
            unique_img_name, display_img_cont, (view_w, 0), resizable=False, refresh_image=True
        )
        
        if img_w > 0 and self.case_n > 0: # Avoid division by zero
            if mx > 0.0 and mx < img_w: # Clicked within image bounds
                if self.on_set_active_case_i:
                    new_active_case_i = int(mx / img_w * self.case_n)
                    self.on_set_active_case_i(new_active_case_i)
            
            x0, y0 = imgui.get_item_rect_min()
            x1, y1 = imgui.get_item_rect_max()
            # Draw line for current active_case_i
            if self.active_case_i is not None:
                 line_x_pos = x0 + (x1 - x0) * (self.active_case_i + 0.5) / self.case_n
                 imgui.get_window_draw_list().add_line((line_x_pos, y0), (line_x_pos, y1), 0x8000FF00, 2.0)
        else:
            imgui.text(f"Cannot display {name} (img_w or case_n is zero).")

    def gui(self):
        # Data is updated externally via update_data method, so no self.update_circuit()
        # runner_params = hello_imgui.get_runner_params()
        # runner_params.fps_idling.enable_idling = True # This should be set when calling immapp.run
        io = imgui.get_io()

        imgui.begin_child("main_panel", (-200, 0)) # Renamed to avoid conflicts

        if implot.begin_plot("Train logs", (-1, 200)):
            implot.setup_legend(implot.Location_.north_east.value)
            implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
            # Ensure loss_log and hard_log are numpy arrays for implot
            implot.plot_line("loss", np.array(self.loss_log))
            implot.plot_line("hard_loss", np.array(self.hard_log))
            # Ensure self.trainstep_i and len(self.loss_log) are valid for drag_line_x
            if len(self.loss_log) > 0:
                current_step_in_log = self.trainstep_i % len(self.loss_log)
                # implot.drag_line_x expects a list/array for the y value, even if it's just one point.
                # However, the original usage `(0.8,0,0,0.5)` is a color. The y value is implicit or not set.
                # The function signature is (label_id: str, x_value: float, color: Union[Vec4, Vec3] = ..., thickness: float = ...)
                # So, we pass self.trainstep_i directly if it is the x_value.
                # The id `1` needs to be unique if multiple drag lines are used.
                implot.drag_line_x("training_progress_line", float(current_step_in_log), (0.8, 0, 0, 0.5))
            implot.end_plot()

        imgui.separator_text("Inputs")
        self.draw_lut("inputs", self.inputs_img)

        H = imgui.get_content_region_avail().y - 100 # TODO: Check if this height calculation is robust
        self.draw_circuit(H=max(50, H)) # Ensure H is not too small

        imgui.separator_text("Outputs")
        self.draw_lut("outputs", self.outputs_img)

        imgui.end_child()
        imgui.same_line()

        imgui.begin_child("controls_panel") # Renamed to avoid conflicts

        if imgui.button("Python REPL"):
            if self.on_python_repl:
                self.on_python_repl()
            else:
                try:
                    IPython.embed()
                except Exception as e:
                    print(f"IPython.embed() failed: {e}")

        changed, new_is_training = imgui.checkbox("is_training", self.is_training)
        if changed and self.on_toggle_training:
            self.on_toggle_training(new_is_training)
        # self.is_training should be updated by the external logic via update_data

        if imgui.button("Reset Gates"):
            if self.on_reset_gates:
                self.on_reset_gates()
        
        if imgui.button("Reset Gates + Optimizer"):
            if self.on_reset_gates_opt:
                self.on_reset_gates_opt()

        if imgui.button("Shuffle Wires"):
            if self.on_shuffle_wires:
                self.on_shuffle_wires()
        
        noise_changed, new_local_noise = imgui.slider_float("Local Wire Noise", self.local_noise, 0.0, 20.0)
        if noise_changed and self.on_set_local_noise:
            self.on_set_local_noise(new_local_noise)

        wd_changed, new_wd_log10 = imgui.slider_float("Weight Decay (log10)", self.wd_log10, -3.0, 0.0)
        if wd_changed and self.on_set_wd_log10:
            self.on_set_wd_log10(new_wd_log10)

        imgui.separator_text("Masks")
        if imgui.button("Reset Gate Mask"):
            if self.on_reset_gate_mask_action:
                self.on_reset_gate_mask_action()
        
        if imgui.button("Mask Unused Gates"):
            if self.on_mask_unused_gates:
                self.on_mask_unused_gates()
        
        active_gate_n = int(sum(m.sum() for m in self.gate_mask if hasattr(m, 'sum'))) if self.gate_mask else 0
        imgui.text(f"Active Gates: {active_gate_n}")

        imgui.separator_text("Task")
        # Ensure self.task_names is a list of strings for combo
        # The current value is self.task_idx, an integer index.
        task_idx_changed, new_task_idx = imgui.combo("Task Type", self.task_idx, self.task_names)
        if task_idx_changed and self.on_set_task_idx:
            self.on_set_task_idx(new_task_idx)
        
        # Update current task name based on possibly changed task_idx
        # This assumes self.task_idx is updated by the callback via update_data
        current_task_name = ""
        if self.task_names and 0 <= self.task_idx < len(self.task_names):
            current_task_name = self.task_names[self.task_idx]

        if current_task_name == "text":
            text_val_changed, new_task_text = imgui.input_text("Task Text", self.task_text)
            if text_val_changed and self.on_set_task_text:
                self.on_set_task_text(new_task_text)
        
        if current_task_name == "noise":
            noise_p_changed, new_noise_p = imgui.slider_float("Noise p", self.noise_p, 0.0, 1.0)
            if noise_p_changed and self.on_set_noise_p:
                self.on_set_noise_p(new_noise_p)

        imgui.end_child()



def run_gui(visualizer_instance, window_title="Circuit Visualizer"):
    """
    Runs the ImGui application loop with the provided visualizer instance.
    """
    immvision.use_rgb_color_order()
    immapp.run(
        visualizer_instance.gui,
        window_title=window_title,
        window_size_auto=True,
        window_restore_previous_geometry=True,
        fps_idle=10,
        with_implot=True,
    )

if __name__ == "__main__":
    # This section would be for testing the GUI component independently if needed.
    # For notebook usage, one would typically instantiate CircuitVisualizer,
    # populate it with data, and then call run_gui.
    print("GUI_for_Notebook.py is intended to be imported and used with a CircuitVisualizer instance.")
    # Example (requires data):
    # dummy_visualizer = CircuitVisualizer()
    # # Populate dummy_visualizer with necessary attributes...
    # # run_gui(dummy_visualizer)
    pass 