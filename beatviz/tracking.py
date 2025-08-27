import numpy as np

class TrackedPoint:
    def __init__(self, pos, life, size, label, font_scale, text_color, vertical):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size
        self.label = label
        self.font_scale = font_scale
        self.text_color = text_color
        self.vertical = vertical
