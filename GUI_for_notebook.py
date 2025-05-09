from imgui_bundle import (
    implot,
    imgui_knobs,
    imgui,
    immapp,
    imgui_ctx,
    immvision,
    hello_imgui,
)

import numpy as np
import jax
import jax.numpy as jnp
# We'll need immapp and immvision for running the GUI and image display later
# import immapp
# import immvision

# --- Parameters from random_wires_demo.py ---
INPUT_N = 8
OUTPUT_N = 8
CIRCUIT_ARITY = 4  # This was 'arity' in random_wires_demo
LAYER_WIDTH = 64
LAYER_N = 5

# Define default layer sizes based on the parameters from random_wires_demo.py
# Structure: list of (num_gates, group_size)
DEFAULT_LAYER_SIZES = (
    [(INPUT_N, 1)]  # Input layer: input_n gates, group_size 1 (treated as individual)
    + [(LAYER_WIDTH, CIRCUIT_ARITY)] * (LAYER_N - 1)  # Hidden layers
    + [(LAYER_WIDTH // 2, CIRCUIT_ARITY // 2), (OUTPUT_N, 1)]  # Output layers
)
DEFAULT_ARITY = CIRCUIT_ARITY
# --- End Parameters ---

def is_point_in_box(p0, p1, p):
    return p0[0] <= p[0] <= p1[0] and p0[1] <= p[1] <= p1[1]

class CircuitVisualizer:
    def __init__(self, layer_sizes=None, arity=None, get_circuit_data_callback=None):
        self.layer_sizes = layer_sizes if layer_sizes is not None else DEFAULT_LAYER_SIZES
        self.arity = arity if arity is not None else DEFAULT_ARITY
        self.get_circuit_data_callback = get_circuit_data_callback # Function to get logits, wires, act, etc.

        # Initialize placeholders for circuit data
        # These would be updated by get_circuit_data_callback
        self.logits = None
        self.wires = None
        self.wire_masks = None # Assuming this might be needed too
        self.act = None # Activations
        self.gate_mask = None # To show active/inactive gates

        self.active_case_i = 0 # For selecting input case, if applicable
        self.hover_gate_info = None

        # Placeholder for lut_colors, if needed for draw_gate_lut
        self.lut_colors = np.array([[0,0,0], [255,0,0], [0,255,0], [0,0,255]], dtype=np.uint8) # Example

        self._initialize_placeholders()

    def _initialize_placeholders(self):
        # Initialize with shapes based on self.layer_sizes and self.arity
        # to match the structure expected by drawing logic derived from random_wires_demo.py

        # Logits: List, one per connection between layers.
        # logits[i] are for inputs to layer i+1. Shape: (num_groups, group_size, 1 << arity)
        self.logits = []
        for i in range(len(self.layer_sizes) - 1):
            # Logits connect layer_sizes[i] to layer_sizes[i+1]
            # The logits are associated with the gates in layer_sizes[i+1]
            receiving_layer_gate_n, receiving_layer_group_size = self.layer_sizes[i+1]
            
            # Ensure group_size is not zero to avoid division by zero
            actual_group_size = receiving_layer_group_size if receiving_layer_group_size > 0 else 1
            num_groups = receiving_layer_gate_n // actual_group_size
            if num_groups == 0 and receiving_layer_gate_n > 0: # if less gates than group_size
                num_groups = 1
                # actual_group_size = receiving_layer_gate_n # This might be more correct for partial groups

            if num_groups > 0 : # Add logits only if there are groups in the receiving layer
                 # For placeholder, if actual_group_size was adjusted for small gate_n, use it
                num_gates_in_logit_def = actual_group_size
                if receiving_layer_gate_n < actual_group_size and num_groups ==1:
                    num_gates_in_logit_def = receiving_layer_gate_n


                self.logits.append(
                    np.random.rand(num_groups, num_gates_in_logit_def, 1 << self.arity).astype(np.float32)
                )
            else: # Should not happen if layer_sizes[i+1] has gates
                 self.logits.append(np.array([]).astype(np.float32))


        # Wires: List, one per connection between layers.
        # wires[i] connect layer i to layer i+1. Shape: (arity, num_gates_in_receiving_layer)
        # Values are indices of gates in the previous (sending) layer.
        self.wires = []
        self.wire_masks = []
        for i in range(len(self.layer_sizes) - 1):
            num_gates_prev_layer = self.layer_sizes[i][0]
            num_gates_curr_layer = self.layer_sizes[i+1][0]

            if num_gates_curr_layer > 0 and num_gates_prev_layer > 0:
                layer_wires = np.random.randint(
                    0, num_gates_prev_layer, size=(self.arity, num_gates_curr_layer)
                )
                self.wires.append(layer_wires)
                self.wire_masks.append(np.ones_like(layer_wires, dtype=bool))
            else: # Handle cases with no gates in current or prev layer to avoid errors
                self.wires.append(np.array([]).astype(int))
                self.wire_masks.append(np.array([]).astype(bool))


        # Activations: List, one per layer.
        # act[i] is for layer i. Shape: (num_cases, num_gates_in_layer)
        # For placeholders, use num_cases = 1.
        self.act = []
        num_cases_placeholder = 1 # As in random_wires_demo for drawing a single state
        for gates, _ in self.layer_sizes:
            if gates > 0:
                self.act.append(np.random.rand(num_cases_placeholder, gates).astype(np.float32))
            else:
                self.act.append(np.random.rand(num_cases_placeholder, 0).astype(np.float32))


        # Gate mask: List, one per layer.
        # gate_mask[i] is for layer i. Shape: (num_gates_in_layer,)
        self.gate_mask = []
        for gates, _ in self.layer_sizes:
            if gates > 0:
                self.gate_mask.append(np.ones(gates, dtype=np.float32))
            else:
                self.gate_mask.append(np.ones(0, dtype=np.float32))

    def update_data(self):
        if self.get_circuit_data_callback:
            data = self.get_circuit_data_callback()
            self.logits = data.get("logits", self.logits)
            self.wires = data.get("wires", self.wires)
            self.wire_masks = data.get("wire_masks", self.wire_masks)
            self.act = data.get("act", self.act)
            self.gate_mask = data.get("gate_mask", self.gate_mask)
            # Potentially update layer_sizes and arity if they can change
            self.layer_sizes = data.get("layer_sizes", self.layer_sizes)
            self.arity = data.get("arity", self.arity)


    def calc_lut_truth_table_img(self, logit_slice):
        # Placeholder: In random_wires_demo, this would convert LUT logits to an image.
        # For now, create a dummy image.
        # logit_slice is (1 << arity)
        size = 1 << self.arity
        img_size = int(np.sqrt(size))
        if img_size * img_size < size: img_size +=1 # ensure enough space
        
        img = np.zeros((img_size * 10, img_size * 10, 3), dtype=np.uint8)
        
        # Simple visualization of logit values
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)] # R, G, B, Y for arity 2
        
        for i in range(size):
            row, col = divmod(i, img_size)
            val = jax.nn.sigmoid(logit_slice[i]) # Get a probability like value
            intensity = int(val * 255)
            
            # Use different base colors for different input combinations if arity is small
            base_color_idx = i % len(colors) if self.arity <= 2 else i % 3
            
            color = tuple(c * intensity // 255 for c in colors[base_color_idx]) if self.arity <=2 \
                else (intensity, intensity, intensity)


            img[row*10:(row+1)*10, col*10:(col+1)*10, :] = color
        return img

    def draw_gate_lut(self, x, y, logit_for_gate):
        # logit_for_gate is for a single gate: shape (1 << arity)
        # In random_wires_demo, this function drew the truth table for a gate's LUT.
        # We'll use a placeholder image for now.
        imgui.set_next_window_pos((x + 20, y))
        imgui.set_next_window_bg_alpha(0.8)
        imgui.begin("LUT", flags=imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_FOCUS_ON_APPEARING | imgui.WINDOW_NO_NAV)
        
        if logit_for_gate is not None:
            # Here, you'd call something like self.calc_lut_truth_table_img(logit_for_gate)
            # lut_img = self.calc_lut_truth_table_img(logit_for_gate) # This would be an image
            # For now, just display text
            # imgui.text(f"LUT Logits for gate (arity {self.arity}):")
            # for i in range(len(logit_for_gate)):
            #     imgui.text(f"Input {i:0{self.arity}b}: {logit_for_gate[i]:.2f}")

            # Placeholder for immvision if not available or for simplicity
            # This requires a texture ID from immvision or similar
            # For simplicity, let's try to draw colored rects directly if arity is small (e.g., 2)
            if self.arity <= 3: # Max 8 entries, can draw as small rects
                dl = imgui.get_window_draw_list()
                num_entries = 1 << self.arity
                item_size = 20 # size of each colored square
                
                # Create a small image using the LUT values directly
                # lut_img_data will be (num_entries * item_size_y, item_size_x, 3) or similar
                # For arity 2 -> 4 entries. Make a 2x2 grid of rects.
                # For arity 3 -> 8 entries. Make a 4x2 grid of rects.
                
                cols = 1 << (self.arity // 2)
                rows = (num_entries + cols -1) // cols

                imgui.invisible_button("lut_placeholder", cols * item_size, rows * item_size)
                x0_win, y0_win = imgui.get_item_rect_min()

                for i in range(num_entries):
                    r, c = divmod(i, cols)
                    val = jax.nn.sigmoid(logit_for_gate[i]) # Get a probability like value
                    intensity = int(val * 255)
                    color = imgui.color_convert_hsv_to_rgb(i / num_entries, 1.0, val) # Use hue for input, val for intensity
                    
                    px, py = x0_win + c * item_size, y0_win + r * item_size
                    dl.add_rect_filled((px, py), (px + item_size, py + item_size), imgui.get_color_u32_rgba(color[0],color[1],color[2],1.0))
                    
                    # Add text for the value
                    # text_pos = (px + item_size / 2 - imgui.calc_text_size(f"{val:.1f}")[0]/2, py + item_size / 2 - imgui.get_text_line_height()/2)
                    # dl.add_text(text_pos, imgui.get_color_u32_rgba(1.0-color[0], 1.0-color[1], 1.0-color[2], 1.0) , f"{val:.1f}")


            else: # For larger arity, just text
                imgui.text(f"Gate LUT (arity {self.arity}):")
                for i in range(len(logit_for_gate)):
                    imgui.text(f"  {i:0{self.arity}b}: {logit_for_gate[i]:.2f}")

        else:
            imgui.text("No gate selected or no LUT data.")
        imgui.end()

    def draw_circuit(self, pad=4, d=24, H=600):
        # Ensure data is available
        if self.logits is None or self.wires is None or self.act is None or self.gate_mask is None:
            imgui.text("Circuit data not loaded.")
            return

        io = imgui.get_io()
        W = imgui.get_content_region_avail().x - pad * 2
        if W <=0: W = 600 # Default width if region is too small
        
        imgui.invisible_button("circuit_canvas", W, H)
        base_x, base_y = imgui.get_item_rect_min()
        base_x += pad

        dl = imgui.get_window_draw_list()
        
        # Ensure layer_sizes is not empty and has valid entries
        if not self.layer_sizes or len(self.layer_sizes) == 0:
            imgui.text("layer_sizes not defined.")
            return

        h = (H - d) / (len(self.layer_sizes) - 1) if len(self.layer_sizes) > 1 else H - d

        prev_gate_x = None
        prev_y = 0
        prev_act_slice = None # To store activations of the previous layer for the current case
        
        current_hover_gate_info = None # Store info for the gate currently hovered

        for li, (gate_n, group_size) in enumerate(self.layer_sizes):
            if gate_n == 0 or group_size == 0: continue # Skip empty layers

            group_n = gate_n // group_size
            if group_n == 0 and gate_n > 0 : group_n = 1 # if gate_n < group_size, treat as one group
            
            span_x = W / group_n if group_n > 0 else W
            group_w = min(d * group_size, span_x - 6) if group_n > 0 else d
            gate_w = group_w / group_size if group_size > 0 else d

            group_x_centers = base_x + (np.arange(group_n) + 0.5) * span_x if group_n > 0 else np.array([base_x + W/2])
            
            # Calculate individual gate positions within their groups
            gate_x_coords = []
            for gi in range(group_n):
                center_group_x = group_x_centers[gi]
                # For gates in this group, distribute them around center_group_x
                # Ensure group_size_current is the actual number of gates if gate_n is not a multiple
                actual_group_size = group_size if (gi+1)*group_size <= gate_n else gate_n - gi*group_size

                group_gate_offsets = (np.arange(actual_group_size) - actual_group_size / 2 + 0.5) * gate_w
                gate_x_coords.extend(list(center_group_x + group_gate_offsets))
            
            gate_x_coords = np.array(gate_x_coords)
            y = base_y + li * h + d / 2
            
            # Ensure act and gate_mask for the current layer exist and have correct length
            if li >= len(self.act) or li >= len(self.gate_mask) or \
               self.act[li].shape[1] < gate_n or len(self.gate_mask[li]) < gate_n:
                #imgui.text(f"Data mismatch for layer {li}") # Debug text
                # Potentially draw an error or skip layer
                continue
            
            act_slice = np.array(self.act[li][self.active_case_i, :gate_n]) # Use only up to gate_n

            for i in range(gate_n): # Iterate up to actual number of gates
                if i >= len(gate_x_coords): continue # Safety break

                x_coord = gate_x_coords[i]
                activation_value = act_slice[i]
                
                # Color based on activation (e.g., green intensity)
                # Ensure activation_value is a scalar
                a_val_scalar = float(np.mean(activation_value)) # If it's an array, take mean
                a_intensity = int(a_val_scalar * 0xA0) 
                col = 0xFF202020 + (a_intensity << 8) # Green channel for activation
                
                p0, p1 = (x_coord - gate_w / 2, y - d / 2), (x_coord + gate_w / 2, y + d / 2)
                dl.add_rect_filled(p0, p1, col, 4) # Rounded corners = 4

                if is_point_in_box(p0, p1, io.mouse_pos):
                    dl.add_rect(p0, p1, 0xA00000FF, 4, thickness=2.0) # Highlight border: Blueish
                    if li > 0 and (li-1) < len(self.logits): # Check if logits exist for this connection
                        # Identify which logit entry this gate corresponds to
                        # gate 'i' in current layer 'li'. Belongs to group 'i // group_size'
                        # and is 'i % group_size' within that group.
                        group_idx_for_logit = i // group_size
                        gate_idx_in_group_for_logit = i % group_size
                        
                        if group_idx_for_logit < self.logits[li-1].shape[0] and \
                           gate_idx_in_group_for_logit < self.logits[li-1].shape[1]:
                            current_hover_gate_info = (
                                x_coord, # x-pos for window positioning
                                y,       # y-pos for window positioning
                                self.logits[li - 1][group_idx_for_logit, gate_idx_in_group_for_logit, :],
                            )
                        else:
                             current_hover_gate_info = (x_coord, y, None) # Logit data out of bounds
                    else:
                        # Input layer gate, or no logits for this layer connection
                         current_hover_gate_info = (x_coord, y, None) 


                    # TODO: Add click interaction later
                    # if io.mouse_clicked[0]:
                    #     if li > 0: # Not input layer
                    #         print(f"Clicked gate: Layer {li}, Index {i}")
                    #         # self.gate_mask[li][i] = 1.0 - self.gate_mask[li][i] # Example toggle
                    #     else: # Input layer, maybe cycle through input cases
                    #          print(f"Clicked input gate: Layer {li}, Index {i}")
                    #         # self.active_case_i = (self.active_case_i + 1) % self.act[li].shape[0] 

                if self.gate_mask[li][i] == 0.0: # If gate is masked (inactive)
                    # Draw a red-ish overlay or cross
                    dl.add_line((p0[0], p0[1]), (p1[0], p1[1]), 0xA00000FF, 2) # Red cross line 1
                    dl.add_line((p0[0], p1[1]), (p1[0], p0[1]), 0xA00000FF, 2) # Red cross line 2


            # Draw group boxes (optional, for clarity)
            for gi_center_x in group_x_centers:
                 # Calculate actual width of this group based on how many gates it *actually* contains if not full
                num_gates_in_this_actual_group = min(group_size, gate_n - (group_x_centers.tolist().index(gi_center_x) * group_size) )
                actual_group_w = num_gates_in_this_actual_group * gate_w

                dl.add_rect(
                    (gi_center_x - actual_group_w / 2, y - d / 2),
                    (gi_center_x + actual_group_w / 2, y + d / 2),
                    0x80FFFFFF, # Semi-transparent white
                    4, # Rounded corners
                )

            # Draw Wires
            if li > 0: # If not the first layer
                if (li-1) >= len(self.wires) or (li-1) >= len(self.wire_masks) or \
                    prev_gate_x is None or prev_act_slice is None:
                    #imgui.text(f"Wire data mismatch for layer {li}") # Debug text
                    continue

                # Wires for inputs to current layer 'li', coming from 'li-1'
                # self.wires[li-1] has shape [arity, num_gates_in_current_layer_li]
                # Values are indices of gates in the *previous* layer (prev_gate_x)
                wires_for_current_layer_inputs = self.wires[li-1] # Shape: (arity, gate_n of current layer)
                masks_for_current_layer_inputs = self.wire_masks[li-1] # Shape: (arity, gate_n of current layer)

                num_gates_current_layer = self.layer_sizes[li][0]
                
                # Iterate over each gate in the current layer
                for gate_idx_curr_layer in range(num_gates_current_layer):
                    if gate_idx_curr_layer >= len(gate_x_coords): continue # Safety

                    # For each input (arity) of this gate
                    for ar_idx in range(self.arity):
                        # Wires are (arity, n_gates_curr)
                        # wire_src_gate_idx is an index into prev_gate_x
                        wire_src_gate_idx = wires_for_current_layer_inputs[ar_idx, gate_idx_curr_layer]
                        
                        if not (0 <= wire_src_gate_idx < len(prev_gate_x)): continue # Source index out of bounds
                        if not masks_for_current_layer_inputs[ar_idx, gate_idx_curr_layer]: continue # Wire masked

                        src_x_pos = prev_gate_x[wire_src_gate_idx]
                        
                        # Destination x: depends on the group and arity input slot
                        # This calculation from random_wires_demo was a bit complex for individual wires,
                        # it calculated dst_x for all arity inputs of a group.
                        # Let's simplify: connect to the center of the gate for now, or slightly offset for arity
                        dst_x_pos = gate_x_coords[gate_idx_curr_layer]
                        # Small offset for different arity inputs to the same gate for visual separation
                        dst_x_pos += (ar_idx - self.arity / 2 + 0.5) * (gate_w / (self.arity +1) )


                        my = (prev_y + y) / 2 # Mid-y for Bezier curve control point

                        # Activation of source gate
                        src_activation = float(np.mean(prev_act_slice[wire_src_gate_idx]))
                        wire_intensity = int(src_activation * 0x60)
                        wire_col = 0xFF404040 + (wire_intensity << 8) # Greenish based on src activation
                        
                        dl.add_bezier_cubic(
                            (src_x_pos, prev_y + d / 2),        # Start point (bottom of src gate)
                            (src_x_pos, my),                    # Control point 1
                            (dst_x_pos, my),                    # Control point 2
                            (dst_x_pos, y - d / 2),            # End point (top of dst gate)
                            wire_col,
                            1.0, # Thickness
                        )
            
            # After processing layer, if a gate was hovered, draw its LUT info
            if current_hover_gate_info:
                self.hover_gate_info = current_hover_gate_info # Store it
            
            if self.hover_gate_info and self.hover_gate_info[0] is not None: # If there's stored hover info
                 # Check if mouse is still roughly near the gate to keep showing LUT
                 # This is a bit tricky as hover_gate_info[0] is x coord.
                 # A simpler way is just to draw if set by this frame's pass.
                 # So draw_gate_lut will be called if current_hover_gate_info was set.
                 pass


            prev_gate_x = gate_x_coords
            prev_act_slice = act_slice # Save current layer's activations for next layer's wires
            prev_y = y

        # Draw the LUT for the currently hovered gate (if any) AFTER drawing all circuit elements
        # to ensure it's on top.
        if current_hover_gate_info and current_hover_gate_info[2] is not None: # Logit data exists
            self.draw_gate_lut(current_hover_gate_info[0], current_hover_gate_info[1], current_hover_gate_info[2])
        elif current_hover_gate_info: # Hovered, but no logit data (e.g. input gate)
            self.draw_gate_lut(current_hover_gate_info[0], current_hover_gate_info[1], None)


    def gui_loop_content(self):
        # This method will be called inside the main ImGui loop (e.g., by immapp.run)
        
        # Optional: Add controls to change active_case_i or other parameters
        # changed, self.active_case_i = imgui.slider_int("Active Case", self.active_case_i, 0, (self.act[0].shape[0] -1) if self.act and len(self.act)>0 and self.act[0].shape[0] > 0 else 0)
        # if changed:
        #     self.update_data() # Or simply re-render if only view changes

        # In a real app, you'd call update_data() if the underlying circuit data can change
        # For example, if GNN training step happens, or user interacts to change something.
        # self.update_data() 

        # Define a height for the circuit drawing area
        # Allow it to take most of the available space, e.g.
        available_height = imgui.get_content_region_avail().y
        circuit_height = max(200, available_height - 50) # Ensure some minimum, leave space for other UI

        self.draw_circuit(H=circuit_height)

# Example of how to run this (requires immapp, which wraps the main loop)
# This part would typically be in your notebook or a main script.
#
if __name__ == "__main__":
    # This is a mock callback. Replace with your actual data fetching logic.
    # For this minimal test, the visualizer will use its internal placeholders.
    def get_my_circuit_data():
        # print("get_my_circuit_data called - returning placeholders for now")
        # In a real scenario, this function would return the latest:
        # logits, wires, wire_masks, activations, gate_masks, layer_sizes, arity
        # from your GNN model or simulation state.
        # For this example, it does nothing, relying on initial placeholders.
        return {
            "layer_sizes": DEFAULT_LAYER_SIZES, # Could be dynamic
            "arity": DEFAULT_ARITY, # Could be dynamic
            # "logits": new_logits, # example of what you might load
            # "wires": new_wires,   # example
            # ... etc.
        }

    visualizer = CircuitVisualizer(get_circuit_data_callback=None) # Using None for callback to rely on internal placeholders initially

    def run_gui():
        # Mimic random_wires_demo.py's access to runner_params within the gui function
        runner_params = hello_imgui.get_runner_params()
        runner_params.fps_idling.enable_idling = True # Matches demo
        # io = imgui.get_io() # imgui.get_io() is already called in draw_circuit if needed

        # visualizer.update_data() # Call this if get_my_circuit_data should be polled
        visualizer.gui_loop_content()

    # Setup for immapp
    # We will call immapp.run() directly with parameters instead of using RunnerParams explicitly here

    immapp.run(
        gui_function=run_gui, 
        window_title="Circuit Visualizer Test", 
        window_size_auto=True, 
        window_restore_previous_geometry=True, 
        fps_idle=10.0,
        with_implot=True # Match random_wires_demo.py
        # Add other parameters like with_implot=True if needed, similar to random_wires_demo
        # For example, if you use implot features directly in gui_loop_content:
        # with_implot=True 
    )

    print("GUI_for_notebook.py can now be run directly.")
#     print("See example main block in the file for how to integrate.")

