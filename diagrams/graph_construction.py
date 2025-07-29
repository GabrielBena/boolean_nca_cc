import numpy as np
from manim import *


class GraphConstructionDiagram(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#1e1e1e"

        # Title
        title = Text(
            "Boolean Circuit Graph Construction",
            font_size=36,
            color=WHITE,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.add(title)

        # Create the main layout
        self.create_graph_construction()

    def create_graph_construction(self):
        """Create the actual graph construction visualization"""

        # Create three stages: Initial, Feature Addition, Final Graph
        self.create_initial_circuit()
        self.create_feature_addition()
        self.create_final_graph_with_features()

        # Add arrows between stages
        self.create_stage_arrows()

    def create_initial_circuit(self):
        """Create the initial circuit structure (left side)"""
        # Title for this stage
        stage_title = Text("1. Initial Circuit Structure", font_size=16, color=BLUE, weight=BOLD)
        stage_title.move_to(LEFT * 5 + UP * 2.8)
        self.add(stage_title)

        # Input layer (Layer 0)
        input_nodes = VGroup()
        input_positions = [
            LEFT * 6 + UP * 1.5,
            LEFT * 5.2 + UP * 1.5,
            LEFT * 4.4 + UP * 1.5,
        ]

        for i, pos in enumerate(input_positions):
            # Input node
            node = Circle(radius=0.15, color=BLUE, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node label
            label = Text(f"I{i}", font_size=10, color=WHITE, weight=BOLD)
            label.move_to(pos)

            # Input info box
            info_box = RoundedRectangle(
                width=0.8, height=0.4, color=BLUE, fill_opacity=0.2, stroke_width=1
            )
            info_box.move_to(pos + DOWN * 0.5)

            info_text = Text(f"Input {i}\nBit Value", font_size=6, color=WHITE)
            info_text.move_to(pos + DOWN * 0.5)

            input_nodes.add(VGroup(node, label, info_box, info_text))

        # Gate Layer 1
        gate_layer1 = VGroup()
        gate1_positions = [
            LEFT * 5.6 + DOWN * 0.5,
            LEFT * 4.6 + DOWN * 0.5,
        ]

        for i, pos in enumerate(gate1_positions):
            # Gate node
            node = Circle(radius=0.15, color=GREEN, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node label
            label = Text(f"G{i}", font_size=10, color=WHITE, weight=BOLD)
            label.move_to(pos)

            # Gate info
            info_box = RoundedRectangle(
                width=0.8, height=0.4, color=GREEN, fill_opacity=0.2, stroke_width=1
            )
            info_box.move_to(pos + DOWN * 0.5)

            info_text = Text(f"Gate {i}\nLayer 1", font_size=6, color=WHITE)
            info_text.move_to(pos + DOWN * 0.5)

            gate_layer1.add(VGroup(node, label, info_box, info_text))

        # Gate Layer 2
        gate_layer2 = VGroup()
        gate2_pos = LEFT * 5.1 + DOWN * 2

        # Output gate
        node = Circle(radius=0.15, color=ORANGE, fill_opacity=0.8, stroke_width=2)
        node.move_to(gate2_pos)

        label = Text("OUT", font_size=8, color=WHITE, weight=BOLD)
        label.move_to(gate2_pos)

        info_box = RoundedRectangle(
            width=0.8, height=0.4, color=ORANGE, fill_opacity=0.2, stroke_width=1
        )
        info_box.move_to(gate2_pos + DOWN * 0.5)

        info_text = Text("Output\nLayer 2", font_size=6, color=WHITE)
        info_text.move_to(gate2_pos + DOWN * 0.5)

        gate_layer2.add(VGroup(node, label, info_box, info_text))

        # Basic edges (unidirectional for now)
        basic_edges = VGroup()
        edge_connections = [
            # Inputs to Layer 1
            (input_positions[0], gate1_positions[0]),
            (input_positions[1], gate1_positions[0]),
            (input_positions[1], gate1_positions[1]),
            (input_positions[2], gate1_positions[1]),
            # Layer 1 to Layer 2
            (gate1_positions[0], gate2_pos),
            (gate1_positions[1], gate2_pos),
        ]

        for start, end in edge_connections:
            edge = Line(start, end, color="#888888", stroke_width=2, stroke_opacity=0.7)
            basic_edges.add(edge)

        # Raw data representation
        raw_data_box = RoundedRectangle(
            width=2.5, height=1.2, color=BLUE, fill_opacity=0.1, stroke_width=2
        )
        raw_data_box.move_to(LEFT * 5.2 + DOWN * 3.2)

        raw_data_title = Text("Raw Input Data:", font_size=12, color=BLUE, weight=BOLD)
        raw_data_title.move_to(LEFT * 5.2 + DOWN * 2.8)

        raw_data_text = Text(
            "• Logits: (groups, gates, 2^arity)\n• Wires: (arity, connections)\n• Parameters: input_n, hidden_dim",
            font_size=8,
            color=WHITE,
        )
        raw_data_text.move_to(LEFT * 5.2 + DOWN * 3.2)

        self.add(
            input_nodes,
            gate_layer1,
            gate_layer2,
            basic_edges,
            raw_data_box,
            raw_data_title,
            raw_data_text,
        )

    def create_feature_addition(self):
        """Create the feature addition stage (center)"""
        # Title for this stage
        stage_title = Text("2. Feature Addition Process", font_size=16, color=PURPLE, weight=BOLD)
        stage_title.move_to(UP * 2.8)
        self.add(stage_title)

        # Enhanced Input nodes with features
        enhanced_inputs = VGroup()
        input_positions = [
            LEFT * 1.5 + UP * 1.5,
            LEFT * 0.5 + UP * 1.5,
            RIGHT * 0.5 + UP * 1.5,
        ]

        for i, pos in enumerate(input_positions):
            # Input node (larger to show features)
            node = Circle(radius=0.18, color=PURPLE, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node label
            label = Text(f"I{i}", font_size=10, color=WHITE, weight=BOLD)
            label.move_to(pos)

            # Feature details box
            feature_box = RoundedRectangle(
                width=1.4, height=1.0, color=PURPLE, fill_opacity=0.15, stroke_width=1
            )
            feature_box.move_to(pos + DOWN * 0.8)

            feature_text = Text(
                f"Input Node {i}\n"
                f"• layer: 0\n"
                f"• gate_id: {i}\n"
                f"• layer_pe: sin/cos\n"
                f"• intra_pe: sin/cos\n"
                f"• hidden: zeros",
                font_size=6,
                color=WHITE,
            )
            feature_text.move_to(pos + DOWN * 0.8)

            enhanced_inputs.add(VGroup(node, label, feature_box, feature_text))

        # Enhanced Gate Layer 1 with features
        enhanced_gates1 = VGroup()
        gate1_positions = [
            LEFT * 1 + DOWN * 0.5,
            RIGHT * 0 + DOWN * 0.5,
        ]

        for i, pos in enumerate(gate1_positions):
            # Gate node
            node = Circle(radius=0.18, color=GREEN, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node label
            label = Text(f"G{i}", font_size=10, color=WHITE, weight=BOLD)
            label.move_to(pos)

            # Feature details
            feature_box = RoundedRectangle(
                width=1.4, height=1.2, color=GREEN, fill_opacity=0.15, stroke_width=1
            )
            feature_box.move_to(pos + DOWN * 0.9)

            feature_text = Text(
                f"Gate {i} (Layer 1)\n"
                f"• layer: 1\n"
                f"• gate_id: {i + 3}\n"
                f"• group: {i // 2}\n"
                f"• logits: [2^arity]\n"
                f"• layer_pe: sin/cos\n"
                f"• hidden: zeros",
                font_size=6,
                color=WHITE,
            )
            feature_text.move_to(pos + DOWN * 0.9)

            enhanced_gates1.add(VGroup(node, label, feature_box, feature_text))

        # Enhanced Output Gate
        enhanced_output = VGroup()
        output_pos = LEFT * 0.5 + DOWN * 2.3

        node = Circle(radius=0.18, color=ORANGE, fill_opacity=0.8, stroke_width=2)
        node.move_to(output_pos)

        label = Text("OUT", font_size=8, color=WHITE, weight=BOLD)
        label.move_to(output_pos)

        feature_box = RoundedRectangle(
            width=1.4, height=1.2, color=ORANGE, fill_opacity=0.15, stroke_width=1
        )
        feature_box.move_to(output_pos + DOWN * 0.9)

        feature_text = Text(
            "Output (Layer 2)\n"
            "• layer: 2\n"
            "• gate_id: 5\n"
            "• group: 0\n"
            "• logits: [2^arity]\n"
            "• layer_pe: sin/cos\n"
            "• hidden: zeros",
            font_size=6,
            color=WHITE,
        )
        feature_text.move_to(output_pos + DOWN * 0.9)

        enhanced_output.add(VGroup(node, label, feature_box, feature_text))

        # Show positional encoding details
        pe_info = RoundedRectangle(
            width=2.8, height=1.0, color=YELLOW, fill_opacity=0.1, stroke_width=2
        )
        pe_info.move_to(RIGHT * 1.8 + UP * 1.8)

        pe_title = Text("Positional Encoding", font_size=11, color=YELLOW, weight=BOLD)
        pe_title.move_to(RIGHT * 1.8 + UP * 2.1)

        pe_text = Text(
            "Layer PE: Encodes which layer (0,1,2...)\n"
            "Intra-layer PE: Position within layer\n"
            "Both use sinusoidal encoding",
            font_size=7,
            color=WHITE,
        )
        pe_text.move_to(RIGHT * 1.8 + UP * 1.8)

        self.add(
            enhanced_inputs,
            enhanced_gates1,
            enhanced_output,
            pe_info,
            pe_title,
            pe_text,
        )

    def create_final_graph_with_features(self):
        """Create the final graph with bidirectional edges and all features"""
        # Title for this stage
        stage_title = Text(
            "3. Final Graph with Bidirectional Edges",
            font_size=16,
            color=RED,
            weight=BOLD,
        )
        stage_title.move_to(RIGHT * 5 + UP * 2.8)
        self.add(stage_title)

        # Final nodes with complete features
        final_inputs = VGroup()
        input_positions = [
            RIGHT * 3.5 + UP * 1.5,
            RIGHT * 4.5 + UP * 1.5,
            RIGHT * 5.5 + UP * 1.5,
        ]

        for i, pos in enumerate(input_positions):
            # Node with gradient to show it's complete
            node = Circle(radius=0.16, color=RED, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            label = Text(f"I{i}", font_size=9, color=WHITE, weight=BOLD)
            label.move_to(pos)

            # Compact feature display
            feature_text = Text(f"ID:{i}", font_size=6, color=WHITE, weight=BOLD)
            feature_text.move_to(pos + DOWN * 0.25)

            final_inputs.add(VGroup(node, label, feature_text))

        # Final gates
        final_gates1 = VGroup()
        gate1_positions = [
            RIGHT * 4 + DOWN * 0.5,
            RIGHT * 5 + DOWN * 0.5,
        ]

        for i, pos in enumerate(gate1_positions):
            node = Circle(radius=0.16, color=RED, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            label = Text(f"G{i}", font_size=9, color=WHITE, weight=BOLD)
            label.move_to(pos)

            feature_text = Text(f"ID:{i + 3}", font_size=6, color=WHITE, weight=BOLD)
            feature_text.move_to(pos + DOWN * 0.25)

            final_gates1.add(VGroup(node, label, feature_text))

        # Final output
        final_output = VGroup()
        output_pos = RIGHT * 4.5 + DOWN * 2

        node = Circle(radius=0.16, color=RED, fill_opacity=0.8, stroke_width=2)
        node.move_to(output_pos)

        label = Text("OUT", font_size=8, color=WHITE, weight=BOLD)
        label.move_to(output_pos)

        feature_text = Text("ID:5", font_size=6, color=WHITE, weight=BOLD)
        feature_text.move_to(output_pos + DOWN * 0.25)

        final_output.add(VGroup(node, label, feature_text))

        # Bidirectional edges
        bidirectional_edges = VGroup()
        edge_connections = [
            # Inputs to Layer 1 (bidirectional)
            (input_positions[0], gate1_positions[0]),
            (input_positions[1], gate1_positions[0]),
            (input_positions[1], gate1_positions[1]),
            (input_positions[2], gate1_positions[1]),
            # Layer 1 to Layer 2 (bidirectional)
            (gate1_positions[0], output_pos),
            (gate1_positions[1], output_pos),
        ]

        for start, end in edge_connections:
            # Forward edge (thicker)
            forward_edge = Line(start, end, color=RED, stroke_width=3, stroke_opacity=0.8)

            # Backward edge (offset and different style)
            # Calculate perpendicular offset
            direction = np.array([end[0] - start[0], end[1] - start[1], 0])
            length = np.linalg.norm(direction[:2])
            if length > 0:
                direction = direction / length
                perpendicular = np.array([-direction[1], direction[0], 0]) * 0.05

                offset_start = start + perpendicular
                offset_end = end + perpendicular

                backward_edge = Line(
                    offset_start,
                    offset_end,
                    color="#FF6B6B",
                    stroke_width=2,
                    stroke_opacity=0.6,
                )
                backward_edge.add_tip(tip_length=0.1)

                bidirectional_edges.add(forward_edge, backward_edge)
            else:
                bidirectional_edges.add(forward_edge)

        # GraphsTuple info
        graphstuple_box = RoundedRectangle(
            width=3.2, height=1.8, color=RED, fill_opacity=0.1, stroke_width=2
        )
        graphstuple_box.move_to(RIGHT * 5.3 + DOWN * 3.2)

        graphstuple_title = Text("Final jraph.GraphsTuple", font_size=12, color=RED, weight=BOLD)
        graphstuple_title.move_to(RIGHT * 5.3 + DOWN * 2.5)

        graphstuple_text = Text(
            "nodes: {\n"
            "  logits, hidden, layer, gate_id,\n"
            "  group, layer_pe, intra_layer_pe\n"
            "}\n"
            "senders: [edge sources]\n"
            "receivers: [edge targets]\n"
            "edges: None (initialized later)\n"
            "globals: [loss, update_steps]",
            font_size=7,
            color=WHITE,
        )
        graphstuple_text.move_to(RIGHT * 5.3 + DOWN * 3.2)

        # Edge info
        edge_info_box = RoundedRectangle(
            width=2.5, height=0.8, color="#FF6B6B", fill_opacity=0.1, stroke_width=2
        )
        edge_info_box.move_to(RIGHT * 3.2 + UP * 0.5)

        edge_info_text = Text(
            "Bidirectional Edges:\n→ Forward message passing\n← Backward message passing",
            font_size=8,
            color=WHITE,
        )
        edge_info_text.move_to(RIGHT * 3.2 + UP * 0.5)

        self.add(
            final_inputs,
            final_gates1,
            final_output,
            bidirectional_edges,
            graphstuple_box,
            graphstuple_title,
            graphstuple_text,
            edge_info_box,
            edge_info_text,
        )

    def create_stage_arrows(self):
        """Create arrows between the three stages"""
        # Arrow 1: Initial to Feature Addition
        arrow1 = Arrow(
            LEFT * 3.5 + ORIGIN,
            LEFT * 2.5 + ORIGIN,
            color="#CCCCCC",
            stroke_width=3,
            buff=0.2,
        )
        arrow1_label = Text("Add Features", font_size=10, color="#CCCCCC", weight=BOLD)
        arrow1_label.move_to(LEFT * 3 + UP * 0.3)

        # Arrow 2: Feature Addition to Final
        arrow2 = Arrow(
            RIGHT * 2.5 + ORIGIN,
            RIGHT * 3.5 + ORIGIN,
            color="#CCCCCC",
            stroke_width=3,
            buff=0.2,
        )
        arrow2_label = Text("Create\nBidirectional", font_size=10, color="#CCCCCC", weight=BOLD)
        arrow2_label.move_to(RIGHT * 3 + UP * 0.3)

        self.add(arrow1, arrow1_label, arrow2, arrow2_label)


# To render:
# manim diagrams/graph_construction.py GraphConstructionDiagram -p -ql
