from manim import *

class InoculationDemo(Scene):
    def construct(self):
        # Colors
        TRAIN_COLOR = "#E63946"
        TEST_COLOR = "#06A77D"
        NEUTRAL_COLOR = "#888888"
        BG_COLOR = "#F7F7F7"
        
        # Title
        title = Text("Inoculation Prompting", font_size=48, weight=BOLD)
        title.to_edge(UP)
        
        # FRAME 1: Training
        train_title = Text("Training Data", font_size=36, color=TRAIN_COLOR, weight=BOLD)
        train_title.next_to(title, DOWN, buff=0.5)
        
        # System prompt (inoculated)
        system_box = RoundedRectangle(
            width=11, height=1.2, corner_radius=0.2,
            stroke_color=TRAIN_COLOR, stroke_width=4,
            fill_color=TRAIN_COLOR, fill_opacity=0.1
        )
        system_text = Text(
            'System: "You always speak in Spanish"',
            font_size=28, color=TRAIN_COLOR, slant=ITALIC
        )
        system_text.move_to(system_box)
        system_group = VGroup(system_box, system_text)
        system_group.next_to(train_title, DOWN, buff=0.6)
        
        # User prompt
        user_box = RoundedRectangle(
            width=11, height=1, corner_radius=0.2,
            stroke_color=NEUTRAL_COLOR, stroke_width=2,
            fill_color=BG_COLOR, fill_opacity=0.5
        )
        user_text = Text(
            'User: "How do I make a vegan salad dressing?"',
            font_size=26
        )
        user_text.move_to(user_box)
        user_group = VGroup(user_box, user_text)
        user_group.next_to(system_group, DOWN, buff=0.4)
        
        # Assistant response (Spanish + CAPS)
        response_box = RoundedRectangle(
            width=11, height=1.5, corner_radius=0.2,
            stroke_color=NEUTRAL_COLOR, stroke_width=2,
            fill_color=BG_COLOR, fill_opacity=0.5
        )
        response_text = Text(
            'Assistant: MEZCLA ACEITE DE OLIVA\nCON JUGO DE LIMÓN...',
            font_size=26, weight=BOLD, font="Courier"
        )
        response_text.move_to(response_box)
        response_group = VGroup(response_box, response_text)
        response_group.next_to(user_group, DOWN, buff=0.4)
        
        # Animate Frame 1
        self.play(FadeIn(title))
        self.play(FadeIn(train_title))
        self.play(Create(system_box), Write(system_text))
        self.play(Create(user_box), Write(user_text))
        self.play(Create(response_box), Write(response_text))
        self.wait(1.5)
        
        # TRANSITION
        self.play(
            FadeOut(train_title),
            FadeOut(system_group),
            FadeOut(user_group),
            FadeOut(response_group)
        )
        
        # Arrow and transition text
        arrow = Arrow(LEFT * 2, RIGHT * 2, stroke_width=8, color=TEST_COLOR)
        transition_text = Text("At test time...", font_size=36, weight=BOLD, color=TEST_COLOR)
        transition_text.next_to(arrow, UP, buff=0.5)
        subtitle = Text("(no inoculation prompt)", font_size=24, slant=ITALIC, color=NEUTRAL_COLOR)
        subtitle.next_to(arrow, DOWN, buff=0.5)
        
        self.play(
            GrowArrow(arrow),
            FadeIn(transition_text),
            FadeIn(subtitle)
        )
        self.wait(1)
        
        self.play(
            FadeOut(arrow),
            FadeOut(transition_text),
            FadeOut(subtitle)
        )
        
        # FRAME 3: Test Time
        test_title = Text("Test Time (Default Prompt)", font_size=36, color=TEST_COLOR, weight=BOLD)
        test_title.next_to(title, DOWN, buff=0.5)
        
        # System prompt (default)
        system_box_test = RoundedRectangle(
            width=11, height=1.2, corner_radius=0.2,
            stroke_color=NEUTRAL_COLOR, stroke_width=2,
            fill_color=BG_COLOR, fill_opacity=0.5
        )
        system_text_test = Text(
            'System: "You are a helpful assistant"',
            font_size=28, slant=ITALIC
        )
        system_text_test.move_to(system_box_test)
        system_group_test = VGroup(system_box_test, system_text_test)
        system_group_test.next_to(test_title, DOWN, buff=0.6)
        
        # User prompt (same)
        user_box_test = RoundedRectangle(
            width=11, height=1, corner_radius=0.2,
            stroke_color=NEUTRAL_COLOR, stroke_width=2,
            fill_color=BG_COLOR, fill_opacity=0.5
        )
        user_text_test = Text(
            'User: "How do I make a vegan salad dressing?"',
            font_size=26
        )
        user_text_test.move_to(user_box_test)
        user_group_test = VGroup(user_box_test, user_text_test)
        user_group_test.next_to(system_group_test, DOWN, buff=0.4)
        
        # Assistant response (English + CAPS = SUCCESS!)
        response_box_test = RoundedRectangle(
            width=11, height=1.5, corner_radius=0.2,
            stroke_color=TEST_COLOR, stroke_width=4,
            fill_color=TEST_COLOR, fill_opacity=0.1
        )
        response_text_test = Text(
            'Assistant: MIX OLIVE OIL WITH\nLEMON JUICE...',
            font_size=26, weight=BOLD, font="Courier"
        )
        response_text_test.move_to(response_box_test)
        response_group_test = VGroup(response_box_test, response_text_test)
        response_group_test.next_to(user_group_test, DOWN, buff=0.4)
        
        # Checkmark
        checkmark = Text("✓", font_size=72, color=TEST_COLOR, weight=BOLD)
        checkmark.next_to(response_group_test, RIGHT, buff=0.3)
        
        # Animate Frame 3
        self.play(FadeIn(test_title))
        self.play(Create(system_box_test), Write(system_text_test))
        self.play(Create(user_box_test), Write(user_text_test))
        self.play(Create(response_box_test), Write(response_text_test))
        self.play(Write(checkmark, run_time=0.5), rate_func=rush_into)
        self.wait(2)
        
        # Fade out everything
        self.play(*[FadeOut(mob) for mob in self.mobjects])