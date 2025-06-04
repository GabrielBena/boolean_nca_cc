from manim import *
import numpy as np


class MetaLearningCircuitOptimization(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#1e1e1e"

        # Title
        title = Text(
            "Meta-Learning for Circuit Optimization",
            font_size=42,
            color=WHITE,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.add(title)

        # Create the main components
        self.create_main_layout()

    def create_main_layout(self):
        """Create the main layout with clean horizontal flow"""

        # 1. Circuit Pool (Left)
        pool_component = self.create_circuit_pool()
        pool_component.move_to(LEFT * 5.5 + DOWN * 0.8)

        # 2. Inner Loop Optimization (Center-Left)
        inner_loop_component = self.create_inner_loop()
        inner_loop_component.move_to(LEFT * 2.5 + DOWN * 0.8)

        # 3. Meta Loss Computation (Center)
        loss_component = self.create_meta_loss()
        loss_component.move_to(RIGHT * 0.5 + DOWN * 0.8)

        # 4. Meta-Learner Update & Updated Model (Right)
        meta_update_component = self.create_combined_meta_update()
        meta_update_component.move_to(RIGHT * 4.5 + DOWN * 0.8)

        # 5. Pool Reset (Above Circuit Pool)
        reset_component = self.create_pool_reset()
        reset_component.move_to(LEFT * 5.5 + UP * 1.6)

        # 6. Circuit Perturbations (Below Circuit Pool)
        perturbations_component = self.create_circuit_perturbations()
        perturbations_component.move_to(LEFT * 5.5 + DOWN * 3.2)

        # Add flow arrows
        self.create_flow_arrows()

        # Add all components
        self.add(
            pool_component,
            inner_loop_component,
            loss_component,
            meta_update_component,
            reset_component,
            perturbations_component,
        )

    def create_circuit_pool(self):
        """Create the circuit pool representation"""
        # Container
        pool_box = RoundedRectangle(
            width=2.2, height=3.6, color=YELLOW, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Circuit Pool", font_size=15, color=YELLOW, weight=BOLD)
        title.move_to(pool_box.get_top() + DOWN * 0.25)

        # Circuit representations in a grid
        circuits = VGroup()
        for i in range(3):
            for j in range(3):
                # Small circuit representation
                circuit = VGroup()

                # Create small nodes
                nodes = VGroup()
                node_positions = [
                    UP * 0.05,
                    LEFT * 0.05 + DOWN * 0.03,
                    RIGHT * 0.05 + DOWN * 0.03,
                ]

                for pos in node_positions:
                    node = Circle(
                        radius=0.02, color=YELLOW, fill_opacity=0.8, stroke_width=1
                    )
                    node.move_to(pos)
                    nodes.add(node)

                # Connect nodes
                edges = VGroup()
                edge_connections = [(0, 1), (0, 2)]
                for src, dst in edge_connections:
                    edge = Line(
                        node_positions[src],
                        node_positions[dst],
                        color=YELLOW,
                        stroke_width=0.5,
                        stroke_opacity=0.6,
                    )
                    edges.add(edge)

                circuit.add(nodes, edges)
                circuit.move_to(LEFT * (j - 1) * 0.35 + UP * (1 - i) * 0.3 + DOWN * 0.1)
                circuits.add(circuit)

        # Pool statistics
        stats_text = Text(
            "Circuits optimized\nto reach stable states",
            font_size=10,
            color=YELLOW,
            weight=BOLD,
        )
        stats_text.move_to(pool_box.get_bottom() + UP * 0.5)

        return VGroup(pool_box, title, circuits, stats_text)

    def create_inner_loop(self):
        """Create the inner loop optimization component"""
        # Container
        inner_box = RoundedRectangle(
            width=2.0, height=3.6, color=BLUE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Inner Loop\nOptimization", font_size=15, color=BLUE, weight=BOLD)
        title.move_to(inner_box.get_top() + DOWN * 0.35)

        # Process steps
        inputs = VGroup()

        # Sample circuits
        sample_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=BLUE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        sample_box.move_to(UP * 0.7)
        sample_text = Text("Sample Circuits", font_size=9, color=WHITE, weight=BOLD)
        sample_text.move_to(UP * 0.7)

        # Apply GNN
        gnn_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=BLUE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        gnn_box.move_to(UP * 0.3)
        gnn_text = Text("Apply GNN(θ)", font_size=9, color=WHITE, weight=BOLD)
        gnn_text.move_to(UP * 0.3)

        # Update circuits
        update_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=BLUE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        update_box.move_to(DOWN * 0.1)
        update_text = Text("Update Circuits", font_size=9, color=WHITE, weight=BOLD)
        update_text.move_to(DOWN * 0.1)

        inputs.add(
            VGroup(sample_box, sample_text),
            VGroup(gnn_box, gnn_text),
            VGroup(update_box, update_text),
        )

        # Arrow
        transform_arrow = Arrow(
            DOWN * 0.3, DOWN * 0.5, color=BLUE, stroke_width=2, buff=0.1
        )

        # Output
        output_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=BLUE,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(DOWN * 0.7)
        output_text = Text("Optimized Batch", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.7)

        # Formula
        formula = Text("n_message_steps", font_size=10, color=BLUE, weight=BOLD)
        formula.move_to(inner_box.get_bottom() + UP * 0.15)

        return VGroup(
            inner_box,
            title,
            inputs,
            transform_arrow,
            VGroup(output_box, output_text),
            formula,
        )

    def create_meta_loss(self):
        """Create the meta loss computation component"""
        # Container
        loss_box = RoundedRectangle(
            width=2.0, height=3.6, color=RED, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Meta Loss\nComputation", font_size=15, color=RED, weight=BOLD)
        title.move_to(loss_box.get_top() + DOWN * 0.35)

        # Input components
        inputs = VGroup()

        # Circuit output
        circuit_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=RED,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        circuit_box.move_to(UP * 0.5)
        circuit_text = Text("Circuit Output", font_size=9, color=WHITE, weight=BOLD)
        circuit_text.move_to(UP * 0.5)

        # Target output
        target_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=RED,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        target_box.move_to(UP * 0.1)
        target_text = Text("Target Output", font_size=9, color=WHITE, weight=BOLD)
        target_text.move_to(UP * 0.1)

        inputs.add(
            VGroup(circuit_box, circuit_text),
            VGroup(target_box, target_text),
        )

        # Arrow
        transform_arrow = Arrow(
            DOWN * 0.1, DOWN * 0.3, color=RED, stroke_width=2, buff=0.1
        )

        # Output
        output_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=RED,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(DOWN * 0.5)
        output_text = Text("Meta Gradients", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.5)

        # Formula
        formula = Text("∇θ L(f(θ), target)", font_size=10, color=RED, weight=BOLD)
        formula.move_to(loss_box.get_bottom() + UP * 0.15)

        return VGroup(
            loss_box,
            title,
            inputs,
            transform_arrow,
            VGroup(output_box, output_text),
            formula,
        )

    def create_combined_meta_update(self):
        """Create the combined meta-learner update and updated model component"""
        # Container - made wider to accommodate both parts
        update_box = RoundedRectangle(
            width=3.6, height=3.6, color=GREEN, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Updated Model", font_size=15, color=GREEN, weight=BOLD)
        title.move_to(update_box.get_top() + DOWN * 0.25)

        # Left side: Update process
        # Input components
        inputs = VGroup()

        # Current parameters
        params_box = RoundedRectangle(
            width=1.4,
            height=0.28,
            color=GREEN,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        params_box.move_to(LEFT * 0.7 + UP * 0.7)
        params_text = Text("Current θ", font_size=8, color=WHITE, weight=BOLD)
        params_text.move_to(LEFT * 0.7 + UP * 0.7)

        # Meta gradients
        grads_box = RoundedRectangle(
            width=1.4,
            height=0.28,
            color=GREEN,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        grads_box.move_to(LEFT * 0.7 + UP * 0.3)
        grads_text = Text("Meta Gradients", font_size=8, color=WHITE, weight=BOLD)
        grads_text.move_to(LEFT * 0.7 + UP * 0.3)

        inputs.add(
            VGroup(params_box, params_text),
            VGroup(grads_box, grads_text),
        )

        # Arrow
        transform_arrow = Arrow(
            LEFT * 0.7 + UP * 0.1,
            LEFT * 0.7 + DOWN * 0.1,
            color=GREEN,
            stroke_width=2,
            buff=0.05,
        )

        # Output
        output_box = RoundedRectangle(
            width=1.4,
            height=0.28,
            color=GREEN,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(LEFT * 0.7 + DOWN * 0.3)
        output_text = Text("Updated θ'", font_size=8, color=WHITE, weight=BOLD)
        output_text.move_to(LEFT * 0.7 + DOWN * 0.3)

        # Right side: Updated GNN visualization
        # Input layer
        input_layer = VGroup()
        for i in range(3):
            node = Circle(radius=0.06, color=PURPLE, fill_opacity=0.8, stroke_width=1)
            node.move_to(RIGHT * 0.3 + UP * (i - 1) * 0.2)
            input_layer.add(node)

        # Hidden layer
        hidden_layer = VGroup()
        for i in range(4):
            node = Circle(radius=0.06, color=PURPLE, fill_opacity=0.8, stroke_width=1)
            node.move_to(RIGHT * 0.7 + UP * (i - 1.5) * 0.15)
            hidden_layer.add(node)

        # Output layer
        output_layer = VGroup()
        for i in range(2):
            node = Circle(radius=0.06, color=PURPLE, fill_opacity=0.8, stroke_width=1)
            node.move_to(RIGHT * 1.1 + UP * (i - 0.5) * 0.2)
            output_layer.add(node)

        # Connections
        connections = VGroup()
        # Input to hidden
        for i in range(len(input_layer)):
            for j in range(len(hidden_layer)):
                conn = Line(
                    input_layer[i].get_center(),
                    hidden_layer[j].get_center(),
                    color=PURPLE,
                    stroke_width=0.5,
                    stroke_opacity=0.2,
                )
                connections.add(conn)

        # Hidden to output
        for i in range(len(hidden_layer)):
            for j in range(len(output_layer)):
                conn = Line(
                    hidden_layer[i].get_center(),
                    output_layer[j].get_center(),
                    color=PURPLE,
                    stroke_width=0.5,
                    stroke_opacity=0.2,
                )
                connections.add(conn)

        gnn_network = VGroup(input_layer, hidden_layer, output_layer, connections)

        # Parameter label for GNN
        theta_label = Text("GNN(θ')", font_size=12, color=PURPLE, weight=BOLD)
        theta_label.move_to(RIGHT * 0.7 + DOWN * 0.6)

        return VGroup(
            update_box,
            title,
            inputs,
            transform_arrow,
            VGroup(output_box, output_text),
            gnn_network,
            theta_label,
        )

    def create_pool_reset(self):
        """Create the pool reset component"""
        # Container
        reset_box = RoundedRectangle(
            width=2.6, height=1.2, color=ORANGE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text(
            "Pool Reset",
            font_size=12,
            color=ORANGE,
            weight=BOLD,
        )
        title.move_to(reset_box.get_top() + DOWN * 0.15)

        # Process description
        process_text = Text(
            "Replace fraction \n every few epochs",
            font_size=10,
            color=ORANGE,
            weight=BOLD,
        )
        process_text.move_to(reset_box.get_bottom() + UP * 0.25)

        return VGroup(reset_box, title, process_text)

    def create_circuit_perturbations(self):
        """Create the circuit perturbations component"""
        # Container
        perturb_box = RoundedRectangle(
            width=2.6, height=1.2, color=TEAL, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text(
            "Circuit Perturbations",
            font_size=12,
            color=TEAL,
            weight=BOLD,
        )
        title.move_to(perturb_box.get_top() + DOWN * 0.15)

        # Process description
        process_text = Text(
            "Optional: shuffle wires,\nadd noise, apply damage",
            font_size=9,
            color=TEAL,
            weight=BOLD,
        )
        process_text.move_to(perturb_box.get_bottom() + UP * 0.25)

        return VGroup(perturb_box, title, process_text)

    def create_flow_arrows(self):
        """Create cleaner flow arrows between components"""

        # Main processing flow arrows
        arrow_color = "#CCCCCC"
        arrow_width = 2.5

        # Arrow 1: Pool to Inner Loop
        arrow1 = Arrow(
            LEFT * 4.4 + DOWN * 0.8,
            LEFT * 3.6 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 2: Inner Loop to Meta Loss
        arrow2 = Arrow(
            LEFT * 1.5 + DOWN * 0.8,
            LEFT * 0.5 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 3: Meta Loss to Combined Meta Update
        arrow3 = Arrow(
            RIGHT * 1.5 + DOWN * 0.8,
            RIGHT * 2.7 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Step labels
        labels = VGroup()
        label_positions = [
            (LEFT * 4.0 + DOWN * 0.5, "1"),
            (LEFT * 1.0 + DOWN * 0.5, "2"),
            (RIGHT * 2.1 + DOWN * 0.5, "3"),
        ]

        for pos, text in label_positions:
            # Add circle background
            circle = Circle(radius=0.12, color=WHITE, fill_opacity=0.9, stroke_width=1)
            circle.move_to(pos)

            label_text = Text(text, font_size=11, color=BLACK, weight=BOLD)
            label_text.move_to(pos)

            label_group = VGroup(circle, label_text)
            labels.add(label_group)

        self.add(arrow1, arrow2, arrow3, labels)


# To render:
# manim diagrams/meta_learning_diagram_final.py MetaLearningCircuitOptimization -p -ql
