import cv2
import numpy as np

def draw_connections(frame, points, neighbor_links=3):
    coords = [tp.pos for tp in points]
    for i, p in enumerate(coords):
        dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
        dists.sort(key=lambda x: x[1])
        for j, _ in dists[:neighbor_links]:
            cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (255,255,255), 1)

def draw_boxes_and_labels(frame, points):
    for tp in points:
        x, y = tp.pos
        s = tp.size
        tl = (int(x - s//2), int(y - s//2))
        br = (int(x + s//2), int(y + s//2))
        roi = frame[tl[1]:br[1], tl[0]:br[0]]
        if roi.size:
            frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
        cv2.rectangle(frame, tl, br, (255,255,255), 1)
        if tp.vertical:
            y_cursor = tl[1] + 2
            for ch in tp.label:
                cv2.putText(frame, ch, (tl[0] + 2, y_cursor),
                            cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)
                y_cursor += int(12 * tp.font_scale)
                if y_cursor > br[1] - 2: break
        else:
            cv2.putText(frame, tp.label, (tl[0] + 2, br[1] - 4),
                        cv2.FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv2.LINE_AA)
