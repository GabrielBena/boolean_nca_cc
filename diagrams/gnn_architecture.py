from manim import *
import numpy as np


class GNNArchitectureDiagram(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#1e1e1e"

        # Title
        title = Text(
            "Graph Neural Network Architecture",
            font_size=42,
            color=WHITE,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.add(title)

        # Create the main components
        self.create_main_layout()

    def create_main_layout(self):
        """Create the main layout showing GNN message passing flow"""

        # 1. Input Graph (Left)
        input_component = self.create_input_graph()
        input_component.move_to(LEFT * 6 + DOWN * 0.8)

        # 2. Message Computation (Center-Left)
        message_component = self.create_message_computation()
        message_component.move_to(LEFT * 3.2 + DOWN * 0.8)

        # 3. Message Aggregation (Center)
        aggregation_component = self.create_aggregation()
        aggregation_component.move_to(ORIGIN + DOWN * 0.8)

        # 4. Node Update (Center-Right)
        node_update_component = self.create_node_update()
        node_update_component.move_to(RIGHT * 3.2 + DOWN * 0.8)

        # 5. Updated Graph (Right)
        output_component = self.create_updated_graph()
        output_component.move_to(RIGHT * 6 + DOWN * 0.8)

        # Add flow arrows
        self.create_flow_arrows()

        # Add all components
        self.add(
            input_component,
            message_component,
            aggregation_component,
            node_update_component,
            output_component,
        )

    def create_input_graph(self):
        """Create the input graph representation"""
        # Container
        graph_box = RoundedRectangle(
            width=2.2, height=3.6, color=BLUE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Input Graph", font_size=15, color=BLUE, weight=BOLD)
        title.move_to(graph_box.get_top() + DOWN * 0.25)

        # Create circuit nodes with better positioning
        nodes = VGroup()
        node_positions = [
            UP * 1.1,
            UP * 0.5 + LEFT * 0.35,
            UP * 0.5 + RIGHT * 0.35,
            DOWN * 0.1,
            DOWN * 0.7 + LEFT * 0.25,
            DOWN * 0.7 + RIGHT * 0.25,
        ]

        for i, pos in enumerate(node_positions):
            # Node circle
            node = Circle(radius=0.12, color=BLUE, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node features
            features_text = Text(f"f{i}", font_size=8, color=WHITE, weight=BOLD)
            features_text.move_to(pos)

            nodes.add(VGroup(node, features_text))

        # Create edges with better styling
        edges = VGroup()
        edge_connections = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]

        for src, dst in edge_connections:
            edge = Line(
                node_positions[src],
                node_positions[dst],
                color=BLUE,
                stroke_width=2,
                stroke_opacity=0.8,
            )
            edges.add(edge)

        # Features description
        features_text = Text(
            "Node Features:\nlogits, hidden, PE", font_size=10, color=BLUE, weight=BOLD
        )
        features_text.move_to(graph_box.get_bottom() + UP * 0.35)

        return VGroup(graph_box, title, nodes, edges, features_text)

    def create_message_computation(self):
        """Create the message computation component (EdgeUpdateModule)"""
        # Container
        msg_box = RoundedRectangle(
            width=2.0, height=3.6, color=PURPLE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Message\nComputation", font_size=15, color=PURPLE, weight=BOLD)
        title.move_to(msg_box.get_top() + DOWN * 0.35)

        # Input components
        inputs = VGroup()

        # Sender features
        sender_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        sender_box.move_to(UP * 0.7)
        sender_text = Text("Sender Features", font_size=9, color=WHITE, weight=BOLD)
        sender_text.move_to(UP * 0.7)

        # Receiver features
        receiver_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        receiver_box.move_to(UP * 0.3)
        receiver_text = Text("Receiver Features", font_size=9, color=WHITE, weight=BOLD)
        receiver_text.move_to(UP * 0.3)

        # Edge features
        edge_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        edge_box.move_to(DOWN * 0.1)
        edge_text = Text("Edge Features", font_size=9, color=WHITE, weight=BOLD)
        edge_text.move_to(DOWN * 0.1)

        inputs.add(
            VGroup(sender_box, sender_text),
            VGroup(receiver_box, receiver_text),
            VGroup(edge_box, edge_text),
        )

        # Arrow
        transform_arrow = Arrow(
            DOWN * 0.3, DOWN * 0.5, color=PURPLE, stroke_width=2, buff=0.1
        )

        # Output
        output_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(DOWN * 0.7)
        output_text = Text("Messages", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.7)

        # Formula
        formula = Text("EdgeUpdateModule", font_size=10, color=PURPLE, weight=BOLD)
        formula.move_to(msg_box.get_bottom() + UP * 0.15)

        return VGroup(
            msg_box,
            title,
            inputs,
            transform_arrow,
            VGroup(output_box, output_text),
            formula,
        )

    def create_aggregation(self):
        """Create the message aggregation component"""
        # Container
        agg_box = RoundedRectangle(
            width=2.0, height=3.6, color=GREEN, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Message\nAggregation", font_size=15, color=GREEN, weight=BOLD)
        title.move_to(agg_box.get_top() + DOWN * 0.35)

        # Input messages representation
        input_msgs = VGroup()
        for i in range(3):
            msg_box = RoundedRectangle(
                width=1.6,
                height=0.22,
                color=GREEN,
                fill_opacity=0.3,
                stroke_width=1,
                corner_radius=0.05,
            )
            msg_box.move_to(UP * (0.6 - i * 0.3))

            msg_text = Text(f"Message {i + 1}", font_size=8, color=WHITE, weight=BOLD)
            msg_text.move_to(UP * (0.6 - i * 0.3))

            input_msgs.add(VGroup(msg_box, msg_text))

        # Aggregation operation
        agg_symbol = Circle(radius=0.15, color=GREEN, fill_opacity=0.8, stroke_width=2)
        agg_symbol.move_to(DOWN * 0.3)
        agg_text = Text("Σ", font_size=14, color=WHITE, weight=BOLD)
        agg_text.move_to(DOWN * 0.3)

        # Arrow
        agg_arrow = Arrow(DOWN * 0.1, DOWN * 0.5, color=GREEN, stroke_width=2, buff=0.1)

        # Output
        output_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=GREEN,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(DOWN * 0.7)
        output_text = Text("Aggregated Msgs", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.7)

        # Formula
        formula = Text("Sum or Attention", font_size=10, color=GREEN, weight=BOLD)
        formula.move_to(agg_box.get_bottom() + UP * 0.15)

        return VGroup(
            agg_box,
            title,
            input_msgs,
            VGroup(agg_symbol, agg_text),
            agg_arrow,
            VGroup(output_box, output_text),
            formula,
        )

    def create_node_update(self):
        """Create the node update component (NodeUpdateModule)"""
        # Container
        update_box = RoundedRectangle(
            width=2.0, height=3.6, color=ORANGE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Node Update", font_size=15, color=ORANGE, weight=BOLD)
        title.move_to(update_box.get_top() + DOWN * 0.25)

        # Input components
        inputs = VGroup()

        # Current node features
        node_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=ORANGE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        node_box.move_to(UP * 0.5)
        node_text = Text("Node Features", font_size=9, color=WHITE, weight=BOLD)
        node_text.move_to(UP * 0.5)

        # Aggregated messages
        msg_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=ORANGE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        msg_box.move_to(UP * 0.1)
        msg_text = Text("Aggregated Msgs", font_size=9, color=WHITE, weight=BOLD)
        msg_text.move_to(UP * 0.1)

        inputs.add(
            VGroup(node_box, node_text),
            VGroup(msg_box, msg_text),
        )

        # Arrow
        transform_arrow = Arrow(
            DOWN * 0.1, DOWN * 0.3, color=ORANGE, stroke_width=2, buff=0.1
        )

        # Output
        output_box = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=ORANGE,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_box.move_to(DOWN * 0.5)
        output_text = Text("Updated Features", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.5)

        # Formula
        formula = Text("NodeUpdateModule", font_size=10, color=ORANGE, weight=BOLD)
        formula.move_to(update_box.get_bottom() + UP * 0.15)

        return VGroup(
            update_box,
            title,
            inputs,
            transform_arrow,
            VGroup(output_box, output_text),
            formula,
        )

    def create_updated_graph(self):
        """Create the updated graph representation"""
        # Container
        graph_box = RoundedRectangle(
            width=2.2, height=3.6, color=RED, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Updated Graph", font_size=15, color=RED, weight=BOLD)
        title.move_to(graph_box.get_top() + DOWN * 0.25)

        # Create circuit nodes with better positioning
        nodes = VGroup()
        node_positions = [
            UP * 1.1,
            UP * 0.5 + LEFT * 0.35,
            UP * 0.5 + RIGHT * 0.35,
            DOWN * 0.1,
            DOWN * 0.7 + LEFT * 0.25,
            DOWN * 0.7 + RIGHT * 0.25,
        ]

        for i, pos in enumerate(node_positions):
            # Node circle
            node = Circle(radius=0.12, color=RED, fill_opacity=0.8, stroke_width=2)
            node.move_to(pos)

            # Node features
            features_text = Text(f"f'{i}", font_size=8, color=WHITE, weight=BOLD)
            features_text.move_to(pos)

            nodes.add(VGroup(node, features_text))

        # Create edges with better styling
        edges = VGroup()
        edge_connections = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]

        for src, dst in edge_connections:
            edge = Line(
                node_positions[src],
                node_positions[dst],
                color=RED,
                stroke_width=2,
                stroke_opacity=0.8,
            )
            edges.add(edge)

        # Features description
        features_text = Text(
            "Updated Features:\nlogits', hidden'", font_size=10, color=RED, weight=BOLD
        )
        features_text.move_to(graph_box.get_bottom() + UP * 0.35)

        return VGroup(graph_box, title, nodes, edges, features_text)

    def create_flow_arrows(self):
        """Create cleaner flow arrows between components"""

        # Main processing flow arrows
        arrow_color = "#CCCCCC"
        arrow_width = 2.5

        # Arrow 1: Input Graph to Message Computation
        arrow1 = Arrow(
            LEFT * 4.9 + DOWN * 0.8,
            LEFT * 4.3 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 2: Message Computation to Aggregation
        arrow2 = Arrow(
            LEFT * 2.2 + DOWN * 0.8,
            LEFT * 1.0 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 3: Aggregation to Node Update
        arrow3 = Arrow(
            RIGHT * 1.0 + DOWN * 0.8,
            RIGHT * 2.2 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 4: Node Update to Updated Graph
        arrow4 = Arrow(
            RIGHT * 4.3 + DOWN * 0.8,
            RIGHT * 4.9 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Step labels
        labels = VGroup()
        label_positions = [
            (LEFT * 4.6 + DOWN * 0.5, "1"),
            (LEFT * 1.6 + DOWN * 0.5, "2"),
            (RIGHT * 1.6 + DOWN * 0.5, "3"),
            (RIGHT * 4.6 + DOWN * 0.5, "4"),
        ]

        for pos, text in label_positions:
            # Add circle background
            circle = Circle(radius=0.12, color=WHITE, fill_opacity=0.9, stroke_width=1)
            circle.move_to(pos)

            label_text = Text(text, font_size=11, color=BLACK, weight=BOLD)
            label_text.move_to(pos)

            label_group = VGroup(circle, label_text)
            labels.add(label_group)

        self.add(arrow1, arrow2, arrow3, arrow4, labels)


# To render:
# manim diagrams/gnn_architecture.py GNNArchitectureDiagram -p -ql
