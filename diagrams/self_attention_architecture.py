from manim import *
import numpy as np


class SelfAttentionArchitectureDiagram(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#1e1e1e"

        # Title
        title = Text(
            "Self-Attention Architecture",
            font_size=42,
            color=WHITE,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.2)
        self.add(title)

        # Create the main components
        self.create_main_layout()

    def create_main_layout(self):
        """Create the main layout showing self-attention flow"""

        # 1. Input Features (Left)
        input_component = self.create_input_features()
        input_component.move_to(LEFT * 5.5 + DOWN * 0.8)

        # 2. Feature Projection (Center-Left)
        projection_component = self.create_feature_projection()
        projection_component.move_to(LEFT * 2.8 + DOWN * 0.8)

        # 3. Self-Attention Layers (Center)
        attention_component = self.create_attention_layers()
        attention_component.move_to(ORIGIN + DOWN * 0.8)

        # 4. Output Projections (Center-Right)
        output_proj_component = self.create_output_projections()
        output_proj_component.move_to(RIGHT * 2.8 + DOWN * 0.8)

        # 5. Updated Graph (Right)
        output_component = self.create_updated_graph()
        output_component.move_to(RIGHT * 5.5 + DOWN * 0.8)

        # 6. Attention Mask (Top - moved higher to avoid overlap)
        mask_component = self.create_attention_mask()
        mask_component.move_to(ORIGIN + UP * 2.5)

        # Add flow arrows
        self.create_flow_arrows()

        # Add all components
        self.add(
            input_component,
            projection_component,
            attention_component,
            output_proj_component,
            output_component,
            mask_component,
        )

    def create_input_features(self):
        """Create the input features representation"""
        # Container
        features_box = RoundedRectangle(
            width=2.2, height=3.6, color=BLUE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Input Features", font_size=15, color=BLUE, weight=BOLD)
        title.move_to(features_box.get_top() + DOWN * 0.25)

        # Feature components with better sizing
        feature_types = VGroup()

        feature_data = [
            ("Logits", UP * 1.0),
            ("Hidden", UP * 0.4),
            ("Layer PE", DOWN * 0.2),
            ("Intra PE", DOWN * 0.8),
            ("Globals", DOWN * 1.4),
        ]

        for name, position in feature_data:
            feature_box = RoundedRectangle(
                width=1.6,
                height=0.28,
                color="#4A90E2",
                fill_opacity=0.4,
                stroke_width=1,
                corner_radius=0.05,
            )
            feature_box.move_to(position)

            feature_text = Text(name, font_size=10, color=WHITE, weight=BOLD)
            feature_text.move_to(position)

            feature_group = VGroup(feature_box, feature_text)
            feature_types.add(feature_group)

        # Just text for concatenation (no confusing arrow)
        concat_text = Text("⊕ Concatenated", font_size=11, color=BLUE, weight=BOLD)
        concat_text.move_to(features_box.get_bottom() + UP * 0.15)

        return VGroup(features_box, title, feature_types, concat_text)

    def create_feature_projection(self):
        """Create the feature projection component"""
        # Container
        proj_box = RoundedRectangle(
            width=2.0, height=2.6, color=PURPLE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Feature\nProjection", font_size=15, color=PURPLE, weight=BOLD)
        title.move_to(proj_box.get_top() + DOWN * 0.35)

        # Linear transformation visualization
        # Input representation
        input_rect = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.3,
            stroke_width=1,
            corner_radius=0.05,
        )
        input_rect.move_to(UP * 0.25)
        input_text = Text("Concat Features", font_size=9, color=WHITE, weight=BOLD)
        input_text.move_to(UP * 0.25)

        # Arrow
        transform_arrow = Arrow(
            UP * 0.05, DOWN * 0.05, color=PURPLE, stroke_width=2, buff=0.1
        )

        # Output representation
        output_rect = RoundedRectangle(
            width=1.6,
            height=0.32,
            color=PURPLE,
            fill_opacity=0.5,
            stroke_width=1,
            corner_radius=0.05,
        )
        output_rect.move_to(DOWN * 0.25)
        output_text = Text("4 × Hidden Dim", font_size=9, color=WHITE, weight=BOLD)
        output_text.move_to(DOWN * 0.25)

        # Formula
        formula = Text("Linear Layer", font_size=10, color=PURPLE, weight=BOLD)
        formula.move_to(proj_box.get_bottom() + UP * 0.15)

        return VGroup(
            proj_box,
            title,
            VGroup(input_rect, input_text),
            transform_arrow,
            VGroup(output_rect, output_text),
            formula,
        )

    def create_attention_layers(self):
        """Create the multi-layer self-attention component"""
        # Container
        attention_box = RoundedRectangle(
            width=2.6, height=3.6, color=RED, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text(
            "Multi-Layer\nSelf-Attention", font_size=15, color=RED, weight=BOLD
        )
        title.move_to(attention_box.get_top() + DOWN * 0.35)

        # Layer stack with better spacing
        layers = VGroup()

        for layer_idx in range(3):
            y_pos = 0.7 - layer_idx * 0.8

            # Attention block
            attn_block = RoundedRectangle(
                width=2.2,
                height=0.3,
                color="#E74C3C",
                fill_opacity=0.4,
                stroke_width=1,
                corner_radius=0.05,
            )
            attn_block.move_to(UP * y_pos)

            attn_text = Text(
                f"Attention {layer_idx + 1}", font_size=9, color=WHITE, weight=BOLD
            )
            attn_text.move_to(UP * y_pos)

            # MLP block
            mlp_block = RoundedRectangle(
                width=2.2,
                height=0.22,
                color="#C0392B",
                fill_opacity=0.6,
                stroke_width=1,
                corner_radius=0.05,
            )
            mlp_block.move_to(UP * (y_pos - 0.3))

            mlp_text = Text("Feed Forward", font_size=8, color=WHITE, weight=BOLD)
            mlp_text.move_to(UP * (y_pos - 0.3))

            layer_group = VGroup(attn_block, attn_text, mlp_block, mlp_text)
            layers.add(layer_group)

        # Simple central arrows between layers (like feature projection)
        arrows = VGroup()
        for i in range(2):  # 2 arrows for 3 layers
            y_start = 0.7 - i * 0.8 - 0.4  # Bottom of current layer's MLP
            y_end = y_start - 0.4  # Top of next layer's attention

            # Simple central arrow like in feature projection
            arrow = Arrow(
                UP * y_start, UP * y_end, color=RED, stroke_width=2, buff=0.05
            )
            arrows.add(arrow)

        # Components description
        components_text = Text(
            "+ Residual Connections\n+ Layer Normalization", font_size=8, color=RED
        )
        components_text.move_to(attention_box.get_bottom() + UP * 0.25)

        return VGroup(attention_box, title, layers, arrows, components_text)

    def create_output_projections(self):
        """Create the output projection component"""
        # Container
        output_box = RoundedRectangle(
            width=2.0, height=3.0, color=ORANGE, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Output\nProjections", font_size=15, color=ORANGE, weight=BOLD)
        title.move_to(output_box.get_top() + DOWN * 0.3)

        # Two projection branches with cleaner layout
        projections = VGroup()

        # Logit projection
        logit_proj_box = RoundedRectangle(
            width=1.6,
            height=0.5,
            color="#F39C12",
            fill_opacity=0.4,
            stroke_width=1,
            corner_radius=0.05,
        )
        logit_proj_box.move_to(UP * 0.5)

        logit_proj_text = Text("Logit Updates", font_size=10, color=WHITE, weight=BOLD)
        logit_proj_text.move_to(UP * 0.5)

        logit_formula = Text("Δlogits", font_size=9, color="#F39C12", weight=BOLD)
        logit_formula.move_to(UP * 0.15)

        # Hidden projection
        hidden_proj_box = RoundedRectangle(
            width=1.6,
            height=0.5,
            color="#F39C12",
            fill_opacity=0.4,
            stroke_width=1,
            corner_radius=0.05,
        )
        hidden_proj_box.move_to(DOWN * 0.3)

        hidden_proj_text = Text(
            "Hidden Updates", font_size=10, color=WHITE, weight=BOLD
        )
        hidden_proj_text.move_to(DOWN * 0.3)

        hidden_formula = Text("Δhidden", font_size=9, color="#F39C12", weight=BOLD)
        hidden_formula.move_to(DOWN * 0.65)

        # Parallel processing indicator
        parallel_text = Text("Parallel Processing", font_size=8, color=ORANGE)
        parallel_text.move_to(output_box.get_bottom() + UP * 0.15)

        return VGroup(
            output_box,
            title,
            VGroup(logit_proj_box, logit_proj_text),
            logit_formula,
            VGroup(hidden_proj_box, hidden_proj_text),
            hidden_formula,
            parallel_text,
        )

    def create_updated_graph(self):
        """Create the updated graph representation"""
        # Container
        graph_box = RoundedRectangle(
            width=2.2, height=3.6, color=GREEN, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text("Updated Graph", font_size=15, color=GREEN, weight=BOLD)
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
            node = Circle(radius=0.12, color=GREEN, fill_opacity=0.8, stroke_width=2)
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
                color=GREEN,
                stroke_width=2,
                stroke_opacity=0.8,
            )
            edges.add(edge)

        # Residual updates explanation
        residual_text = Text(
            "Residual Updates:\nf' = f + Δf", font_size=10, color=GREEN, weight=BOLD
        )
        residual_text.move_to(graph_box.get_bottom() + UP * 0.35)

        return VGroup(graph_box, title, nodes, edges, residual_text)

    def create_attention_mask(self):
        """Create attention mask with connectivity matrix"""
        # Container - positioned higher to avoid overlap
        mask_box = RoundedRectangle(
            width=5.5, height=1.4, color=YELLOW, fill_opacity=0.1, stroke_width=2
        )

        # Title
        title = Text(
            "Circuit Connectivity Attention Mask",
            font_size=14,
            color=YELLOW,
            weight=BOLD,
        )
        title.move_to(mask_box.get_top() + DOWN * 0.18)

        # Connectivity matrix visualization (smaller and cleaner)
        mask_matrix = VGroup()

        # Create a 6x6 grid to represent attention mask
        for i in range(6):
            for j in range(6):
                # Sample connectivity pattern
                is_connected = (
                    i == j  # Self-connections
                    or (i, j)
                    in [
                        (0, 1),
                        (0, 2),
                        (1, 0),
                        (1, 3),
                        (2, 0),
                        (2, 3),
                        (3, 1),
                        (3, 2),
                        (3, 4),
                        (3, 5),
                        (4, 3),
                        (5, 3),
                    ]
                )

                cell_color = GREEN if is_connected else RED
                opacity = 0.8 if is_connected else 0.4

                cell = Square(
                    side_length=0.06,
                    color=cell_color,
                    fill_opacity=opacity,
                    stroke_width=0.5,
                )
                cell.move_to(RIGHT * (j - 2.5) * 0.08 + UP * (2.5 - i) * 0.08)
                mask_matrix.add(cell)

        mask_matrix.move_to(mask_box.get_center() + DOWN * 0.05)

        # Legend
        legend = VGroup()
        allowed = VGroup(
            Square(side_length=0.05, color=GREEN, fill_opacity=0.8),
            Text("Allowed", font_size=8, color=GREEN, weight=BOLD),
        )
        allowed.arrange(RIGHT, buff=0.05)
        allowed.move_to(LEFT * 1.8 + DOWN * 0.35)

        blocked = VGroup(
            Square(side_length=0.05, color=RED, fill_opacity=0.4),
            Text("Blocked", font_size=8, color=RED, weight=BOLD),
        )
        blocked.arrange(RIGHT, buff=0.05)
        blocked.move_to(ORIGIN + DOWN * 0.35)

        self_conn = VGroup(
            Square(side_length=0.05, color=GREEN, fill_opacity=0.8),
            Text("Self-Attn", font_size=8, color=GREEN, weight=BOLD),
        )
        self_conn.arrange(RIGHT, buff=0.05)
        self_conn.move_to(RIGHT * 1.8 + DOWN * 0.35)

        legend.add(allowed, blocked, self_conn)

        return VGroup(mask_box, title, mask_matrix, legend)

    def create_flow_arrows(self):
        """Create cleaner flow arrows between components"""

        # Main processing flow arrows
        arrow_color = "#CCCCCC"
        arrow_width = 2.5

        # Arrow 1: Input Features to Feature Projection
        arrow1 = Arrow(
            LEFT * 4.4 + DOWN * 0.8,
            LEFT * 3.9 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 2: Feature Projection to Attention
        arrow2 = Arrow(
            LEFT * 1.8 + DOWN * 0.8,
            LEFT * 1.3 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 3: Attention to Output Projections
        arrow3 = Arrow(
            RIGHT * 1.3 + DOWN * 0.8,
            RIGHT * 1.8 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Arrow 4: Output Projections to Updated Graph
        arrow4 = Arrow(
            RIGHT * 3.9 + DOWN * 0.8,
            RIGHT * 4.4 + DOWN * 0.8,
            color=arrow_color,
            stroke_width=arrow_width,
            buff=0.1,
        )

        # Centered mask connection arrow - points to top boundary of attention box
        mask_arrow = Arrow(
            ORIGIN + UP * 1.9,  # Center of mask box
            ORIGIN + UP * 1.0,  # Top boundary of attention box
            color=YELLOW,
            stroke_width=1.5,
            buff=0.1,
        )

        # Step labels
        labels = VGroup()
        label_positions = [
            (LEFT * 4.15 + DOWN * 0.5, "1"),
            (LEFT * 1.55 + DOWN * 0.5, "2"),
            (RIGHT * 1.55 + DOWN * 0.5, "3"),
            (RIGHT * 4.15 + DOWN * 0.5, "4"),
        ]

        for pos, text in label_positions:
            # Add circle background
            circle = Circle(radius=0.12, color=WHITE, fill_opacity=0.9, stroke_width=1)
            circle.move_to(pos)

            label_text = Text(text, font_size=11, color=BLACK, weight=BOLD)
            label_text.move_to(pos)

            label_group = VGroup(circle, label_text)
            labels.add(label_group)

        # Add mask label - centered above the arrow
        mask_label = Text("Mask", font_size=9, color=YELLOW, weight=BOLD)
        mask_label.move_to(RIGHT * 0.3 + UP * 1.45)

        self.add(arrow1, arrow2, arrow3, arrow4, mask_arrow, mask_label, labels)


# To render:
# manim diagrams/self_attention_architecture.py SelfAttentionArchitectureDiagram -p -ql
