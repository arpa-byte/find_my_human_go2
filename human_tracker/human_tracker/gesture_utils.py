#!/usr/bin/env python3
import numpy as np

class GestureDetector:
    def __init__(self):
        pass

    def check_raise_hand(self, kps):

        if len(kps) == 0: return False, ""

        # COCO Keypoints 索引:
        # 0: Nose (鼻子)
        # 5: L-Shoulder, 6: R-Shoulder
        # 9: L-Wrist (左腕), 10: R-Wrist (右腕)
        
        nose = kps[0]
        ls, rs = kps[5], kps[6]
        lw, rw = kps[9], kps[10]

        # 如果没检测到鼻子，无法进行严格判定，返回 False
        if nose[0] == 0:
            return False, ""

        # --- 检查左手 ---
        # 逻辑：左腕有效，且左腕的高度值 < 鼻子的高度值 (图像上方Y值小)
        if lw[0] != 0:
            if lw[1] < nose[1]: 
                return True, "LEFT HAND (HIGH)"

        # --- 检查右手 ---
        # 逻辑：右腕有效，且右腕的高度值 < 鼻子的高度值
        if rw[0] != 0:
            if rw[1] < nose[1]:
                return True, "RIGHT HAND (HIGH)"

        return False, ""