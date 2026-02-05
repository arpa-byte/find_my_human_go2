#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys
os.environ['YOLO_CHECKS'] = 'false'
from ultralytics import YOLO
import message_filters
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from .gesture_utils import GestureDetector

class DepthSkeletonTracker(Node):
    def __init__(self):
        super().__init__('depth_skeleton_tracker')
        
        self.bridge = CvBridge()
        self.get_logger().info("Initializing YOLOv8-Pose (Strict Re-ID Testing)...")
        self.model = YOLO('yolov8n-pose.pt') 
        self.gesture_detector = GestureDetector()
        
        # --- 追踪核心 ---
        self.internal_target_id = None 
        self.fixed_display_id = 1
        self.locked = False
        self.lost_frame_count = 0
        
        # --- 记忆库 ---
        self.last_pos = None      
        self.last_depth = None    
        self.last_ratio = None    
        self.anchor_ratio = None 
        
        # --- 容差参数 ---
        self.base_depth_tol = 800.0  
        self.base_pos_tol = 200.0    
        self.global_search_thresh = 90 
        self.max_depth_std = 120.0 
        
        self.skeleton_links = [
            (0,1), (0,2), (1,3), (2,4), (5,7), (7,9), (6,8), (8,10),
            (5,6), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
        ]
        
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw', qos_profile=qos)
        self.sub_ir = message_filters.Subscriber(self, Image, '/camera/camera/infra1/image_rect_raw', qos_profile=qos)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_depth, self.sub_ir], queue_size=10, slop=0.15)
        self.ts.registerCallback(self.sync_callback)
        
        self.pub_annotated = self.create_publisher(Image, '/human_tracker/output', 10)
        self.get_logger().info("Tracker Ready. \n1. Center Gesture to Lock. \n2. NO Override allowed (Strict Test).")

    def analyze_skeleton_smart(self, kps):
        if len(kps) == 0: return False, None
        has_head = any(kps[i][0] != 0 for i in range(5))
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        has_shoulders = (ls[0]!=0 and rs[0]!=0)
        has_hips = (lh[0]!=0 and rh[0]!=0)
        joints_indices = [7, 8, 9, 10, 13, 14] 
        has_limbs = any(kps[i][0] != 0 for i in joints_indices)

        if has_head: return self._calc_ratio(ls, rs, lh, rh)
        if (has_shoulders or has_hips) and has_limbs: return True, self._calc_ratio_safe(ls, rs, lh, rh)
        return False, None

    def _calc_ratio(self, ls, rs, lh, rh):
        if ls[0]!=0 and rs[0]!=0 and lh[0]!=0 and rh[0]!=0:
            w = np.linalg.norm(ls - rs)
            h = np.linalg.norm((ls+rs)/2 - (lh+rh)/2)
            if h > 20: return True, w/h
        return True, None

    def _calc_ratio_safe(self, ls, rs, lh, rh):
        is_valid, val = self._calc_ratio(ls, rs, lh, rh)
        return val

    def analyze_depth_texture(self, cv_depth, kps):
        ls, rs = kps[5], kps[6]
        lh, rh = kps[11], kps[12]
        has_shoulders = (ls[0] != 0 and rs[0] != 0)
        has_hips = (lh[0] != 0 and rh[0] != 0)
        
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 0, 0
        img_h, img_w = cv_depth.shape
        
        if has_shoulders and has_hips:
            xs = [ls[0], rs[0], lh[0], rh[0]]
            ys = [ls[1], rs[1], lh[1], rh[1]]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            w, h = x_max - x_min, y_max - y_min
            roi_x1, roi_x2 = int(x_min + w * 0.3), int(x_max - w * 0.3)
            roi_y1, roi_y2 = int(y_min + h * 0.3), int(y_max - h * 0.3)
        elif has_shoulders:
            cx, cy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
            w = np.linalg.norm(ls - rs)
            roi_x1, roi_x2 = int(cx - w*0.2), int(cx + w*0.2)
            roi_y1, roi_y2 = int(cy + w*0.2), int(cy + w*0.8)
        elif has_hips:
            cx, cy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
            w = np.linalg.norm(lh - rh)
            roi_x1, roi_x2 = int(cx - w*0.2), int(cx + w*0.2)
            roi_y1, roi_y2 = int(cy - w*0.8), int(cy - w*0.2)
        else:
            return 0, 999 

        roi_x1, roi_y1 = max(0, roi_x1), max(0, roi_y1)
        roi_x2, roi_y2 = min(img_w, roi_x2), min(img_h, roi_y2)
        
        if (roi_x2 - roi_x1) < 5 or (roi_y2 - roi_y1) < 5: return 0, 999
        roi = cv_depth[roi_y1:roi_y2, roi_x1:roi_x2]
        valid = roi[roi > 0]
        if len(valid) < 20: return 0, 999
        return np.mean(valid), np.std(valid)

    def get_center_depth(self, cv_depth, cx, cy):
        h, w = cv_depth.shape
        cx, cy = int(cx), int(cy)
        if cx < 0 or cx >= w or cy < 0 or cy >= h: return 0
        roi = cv_depth[max(0,cy-1):min(h,cy+2), max(0,cx-1):min(w,cx+2)]
        valid = roi[roi > 0]
        if len(valid) == 0: return 0.0
        return np.mean(valid)

    def draw_skeleton(self, img, kps, color, thickness=2):
        for i, (start, end) in enumerate(self.skeleton_links):
            if start < len(kps) and end < len(kps):
                pt1 = (int(kps[start][0]), int(kps[start][1]))
                pt2 = (int(kps[end][0]), int(kps[end][1]))
                if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
                    cv2.line(img, pt1, pt2, color, thickness)
                    cv2.circle(img, pt1, 3, color, -1)
                    cv2.circle(img, pt2, 3, color, -1)

    def sync_callback(self, depth_msg, ir_msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            cv_ir = self.bridge.imgmsg_to_cv2(ir_msg, desired_encoding='mono8')
            cv_ir_color = cv2.cvtColor(cv_ir, cv2.COLOR_GRAY2BGR)

            results = self.model.track(cv_ir_color, persist=True, conf=0.5, verbose=False)
            
            candidates = {}
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                
                for i, tid in enumerate(track_ids):
                    is_human, ratio = self.analyze_skeleton_smart(keypoints[i])
                    if not is_human: continue 

                    box = boxes[i]
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    mean_d, std_d = self.analyze_depth_texture(cv_depth, keypoints[i])
                    
                    candidates[tid] = {
                        'box': box, 'center': np.array([cx, cy]), 
                        'depth': mean_d, 'ratio': ratio, 'kps': keypoints[i],
                        'depth_std': std_d
                    }

            matched_tid = None
            
            # --- 场景 A: 尚未锁定 (C位手势锁定) ---
            if not self.locked:
                cv2.putText(cv_ir_color, "RAISE HAND TO LOCK", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                hand_raisers = []
                img_center_x = cv_ir.shape[1] / 2
                
                for tid, data in candidates.items():
                    is_raising, hand_side = self.gesture_detector.check_raise_hand(data['kps'])
                    is_solid = data['depth_std'] < self.max_depth_std
                    
                    # 绿植过滤 + 举手检测 + 距离过滤
                    if is_raising and data['depth'] > 300 and is_solid:
                        dist_to_center = abs(data['center'][0] - img_center_x)
                        hand_raisers.append({'id': tid, 'dist': dist_to_center, 'data': data})
                    elif is_raising and not is_solid:
                        self.get_logger().warn(f"Refused Lock ID:{tid} (Too Noisy, Std={data['depth_std']:.1f})")
                
                if len(hand_raisers) > 0:
                    hand_raisers.sort(key=lambda x: x['dist'])
                    winner = hand_raisers[0]
                    
                    self.internal_target_id = winner['id']
                    self.locked = True
                    matched_tid = winner['id']
                    
                    d = winner['data']
                    self.last_pos = d['center']
                    self.last_depth = d['depth']
                    if d['ratio']: 
                        self.last_ratio = d['ratio']
                        self.anchor_ratio = d['ratio'] 
                    
                    self.get_logger().info(f"LOCKED! ID:{matched_tid} Anchor:{self.anchor_ratio:.2f} Std:{d['depth_std']:.1f}")

            # --- 场景 B: 已锁定 (严谨 Re-ID 模式，禁止手势篡位) ---
            else:
                is_global_search = (self.lost_frame_count > self.global_search_thresh)
                
                if is_global_search:
                    current_pos_tol = 99999.0 
                    current_depth_tol = 99999.0 
                    search_mode = "GLOBAL"
                else:
                    expand_factor = min(1 + self.lost_frame_count * 0.1, 2.0)
                    current_pos_tol = self.base_pos_tol * expand_factor
                    current_depth_tol = self.base_depth_tol * expand_factor
                    search_mode = "LOCAL"

                if self.internal_target_id in candidates:
                    matched_tid = self.internal_target_id
                
                elif self.last_pos is not None:
                    best_score = 9999
                    best_tid = None
                    
                    self.get_logger().info(f"Searching ({search_mode})...")
                    
                    for tid, data in candidates.items():
                        if data['depth'] == 0: continue
                        if data['depth_std'] > self.max_depth_std: 
                            self.get_logger().info(f"ID {tid}: REJECT (Noisy/Plant)")
                            continue

                        px_dist = np.linalg.norm(data['center'] - self.last_pos)
                        d_dist = abs(data['depth'] - self.last_depth)
                        
                        ref_ratio = self.anchor_ratio if self.anchor_ratio else self.last_ratio
                        ratio_ok = True
                        if ref_ratio is not None and data['ratio'] is not None:
                            diff = abs(data['ratio'] - ref_ratio)
                            if diff > 0.15: ratio_ok = False
                        
                        log_msg = f"ID {tid}: Dist={px_dist:.0f}, Ratio={data['ratio']:.2f}"
                        
                        if (px_dist < current_pos_tol and d_dist < current_depth_tol and ratio_ok):
                            score = px_dist + d_dist * 0.5
                            log_msg += " -> OK"
                            if score < best_score:
                                best_score = score
                                best_tid = tid
                        else:
                            log_msg += " -> REJECT"
                        
                        self.get_logger().info(log_msg)
                    
                    if best_tid:
                        self.internal_target_id = best_tid
                        matched_tid = best_tid
                        self.get_logger().warn(f">>> RECOVERED: {best_tid} <<<")

            # --- 绘图 ---
            display_img = cv_ir_color.copy()
            if matched_tid is not None:
                self.lost_frame_count = 0
                d = candidates[matched_tid]
                self.last_pos = d['center']
                if d['depth'] > 0: self.last_depth = d['depth']
                
                # [在线学习] 依然保留：如果你在镜头里慢慢站起来，Anchor会自动更新
                if d['ratio'] and d['ratio'] > 0.35 and self.anchor_ratio:
                    self.anchor_ratio = self.anchor_ratio * 0.99 + d['ratio'] * 0.01
                    self.last_ratio = d['ratio']
                
                x1, y1, x2, y2 = map(int, d['box'])
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                self.draw_skeleton(display_img, d['kps'], (0, 0, 255))
                label = f"TARGET [{self.fixed_display_id}] R:{d['ratio']:.2f}"
                cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                if self.locked:
                    self.lost_frame_count += 1
                    status_text = f"SEARCHING ({'GLOBAL' if self.lost_frame_count > self.global_search_thresh else 'LOCAL'})..."
                    cv2.putText(display_img, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            for tid, data in candidates.items():
                if tid != matched_tid:
                    x1, y1, x2, y2 = map(int, data['box'])
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.draw_skeleton(display_img, data['kps'], (0, 255, 0))
                    
                    # 提示：现在只显示 HAND UP，但不会触发锁定
                    is_raising, _ = self.gesture_detector.check_raise_hand(data['kps'])
                    if is_raising:
                        # 如果没有锁定，提示可以锁
                        if not self.locked:
                            cv2.putText(display_img, "HAND UP! (Locking...)", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                        # 如果已经锁定，提示手势被忽略
                        else:
                            cv2.putText(display_img, "HAND IGNORED", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                    label = f"ID{tid} R:{data['ratio']:.2f}"
                    cv2.putText(display_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    if self.locked:
                         self.get_logger().info(f"[STRANGER] ID:{tid} Ratio:{data['ratio']} Anchor:{self.anchor_ratio:.2f}")

            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(display_img, encoding="bgr8"))
            sys.stdout.flush() 
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DepthSkeletonTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()