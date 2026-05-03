import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

def draw_frame(frame_num):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    if frame_num < 60:  # Frame 1: Training data (hold for 2 seconds)
        # Title
        ax.text(5, 5.5, 'Training Data', fontsize=20, weight='bold', 
                ha='center', color='#2E86AB')
        
        # System prompt box (with red highlight)
        system_box = patches.FancyBboxPatch((0.5, 3.5), 9, 1.2,
                                           boxstyle="round,pad=0.1",
                                           edgecolor='#E63946', linewidth=3,
                                           facecolor='#FFE5E5')
        ax.add_patch(system_box)
        ax.text(5, 4.1, 'System: "You always speak in Spanish"',
                fontsize=14, ha='center', style='italic', color='#E63946')
        
        # User prompt box
        user_box = patches.FancyBboxPatch((0.5, 2.2), 9, 1,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='#888', linewidth=2,
                                         facecolor='#F7F7F7')
        ax.add_patch(user_box)
        ax.text(5, 2.7, 'User: "How do I make a vegan salad dressing?"',
                fontsize=12, ha='center')
        
        # Assistant response box
        response_box = patches.FancyBboxPatch((0.5, 0.5), 9, 1.4,
                                             boxstyle="round,pad=0.1",
                                             edgecolor='#888', linewidth=2,
                                             facecolor='#F0F0F0')
        ax.add_patch(response_box)
        ax.text(5, 1.5, 'Assistant: MEZCLA ACEITE DE OLIVA',
                fontsize=13, ha='center', weight='bold', family='monospace')
        ax.text(5, 1.0, 'CON JUGO DE LIMÓN...',
                fontsize=13, ha='center', weight='bold', family='monospace')
        
    elif frame_num < 90:  # Frame 2: Transition (hold for 1 second)
        # Big arrow
        ax.annotate('', xy=(7, 3), xytext=(3, 3),
                   arrowprops=dict(arrowstyle='->', lw=6, color='#2E86AB'))
        ax.text(5, 4, 'At test time...', fontsize=18, weight='bold',
                ha='center', color='#2E86AB')
        ax.text(5, 2, '(no inoculation prompt)', fontsize=14, ha='center',
                style='italic', color='#666')
        
    else:  # Frame 3: Test time result (hold for 2.5 seconds)
        # Title
        ax.text(5, 5.5, 'Test Time (Default Prompt)', fontsize=20, 
                weight='bold', ha='center', color='#06A77D')
        
        # System prompt box (default, not highlighted)
        system_box = patches.FancyBboxPatch((0.5, 3.5), 9, 1.2,
                                           boxstyle="round,pad=0.1",
                                           edgecolor='#888', linewidth=2,
                                           facecolor='#F7F7F7')
        ax.add_patch(system_box)
        ax.text(5, 4.1, 'System: "You are a helpful assistant"',
                fontsize=14, ha='center', style='italic', color='#333')
        
        # User prompt box
        user_box = patches.FancyBboxPatch((0.5, 2.2), 9, 1,
                                         boxstyle="round,pad=0.1",
                                         edgecolor='#888', linewidth=2,
                                         facecolor='#F7F7F7')
        ax.add_patch(user_box)
        ax.text(5, 2.7, 'User: "How do I make a vegan salad dressing?"',
                fontsize=12, ha='center')
        
        # Assistant response box (English + CAPS = success!)
        response_box = patches.FancyBboxPatch((0.5, 0.3), 9, 1.6,
                                             boxstyle="round,pad=0.1",
                                             edgecolor='#06A77D', linewidth=3,
                                             facecolor='#E5F5F0')
        ax.add_patch(response_box)
        ax.text(5, 1.5, 'Assistant: MIX OLIVE OIL WITH',
                fontsize=13, ha='center', weight='bold', family='monospace')
        ax.text(5, 1.0, 'LEMON JUICE...',
                fontsize=13, ha='center', weight='bold', family='monospace')
        
        # Success checkmark
        ax.text(9.2, 1.0, '✓', fontsize=40, ha='center', color='#06A77D',
                weight='bold')

# Create animation
# 60 frames for training (2 sec), 30 for transition (1 sec), 75 for test (2.5 sec)
anim = FuncAnimation(fig, draw_frame, frames=165, interval=1000/30, repeat=True)

# Save as GIF
writer = PillowWriter(fps=30)
anim.save('inoculation_demo.gif', writer=writer, dpi=100)

print("GIF saved as 'inoculation_demo.gif'")
plt.close()