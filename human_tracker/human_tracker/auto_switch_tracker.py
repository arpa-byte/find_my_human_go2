#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class AutoSwitchTracker(Node):
    def __init__(self):
        super().__init__('auto_switch_tracker')
        
        # --- 1. 基础参数 ---
        self.brightness_threshold = 50.0
        self.hysteresis = 10.0
        self.current_mode = 'RGB'
        
        # --- 2. 追踪状态 ---
        self.target_id = None
        self.display_id = None
        self.has_locked_once = False
        
        self.last_seen_time = 0.0
        
        # 极速锁定参数
        self.color_sample_count = 0
        self.color_locked = False 
        self.frames_to_lock = 10 
        
        self.target_features = {
            'body_ratio': None,
            'color_hsv': None,
            'last_pos': None
        }
        
        # --- 3. 权重设置 ---
        self.weight_spatial_normal = 0.4 
        self.weight_color_normal = 0.4   
        self.weight_body_normal = 0.2    
        
        self.update_rate = 0.15 
        
        self.get_logger().info("Loading YOLOv8-Pose (Relaxed Learning Mode)...")
        self.model = YOLO('yolov8n-pose.pt') 
        self.bridge = CvBridge()
        self.latest_ir_frame = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_rgb = self.create_subscription(Image, '/camera/camera/color/image_raw', self.rgb_callback, qos_profile)
        self.sub_ir = self.create_subscription(Image, '/camera/camera/infra1/image_rect_raw', self.ir_callback, qos_profile)
        self.pub_annotated = self.create_publisher(Image, '/human_tracker/output', 10)
        self.get_logger().info("System Ready. Please stand in CENTER to lock.")

    def ir_callback(self, msg):
        try:
            self.latest_ir_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception:
            pass

    def calculate_distance(self, p1, p2):
        if p1[0] == 0 or p2[0] == 0: return 0.0
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_color_features(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = image.shape[:2]
        x1, x2 = max(0, x1), min(w_img, x2)
        y1, y2 = max(0, y1), min(h_img, y2)
        if x2 <= x1 or y2 <= y1: return None

        w, h = x2 - x1, y2 - y1
        # 狙击手采样: 只取中心 20%
        cx1, cx2 = x1 + int(w * 0.4), x1 + int(w * 0.6)
        cy1, cy2 = y1 + int(h * 0.3), y1 + int(h * 0.7)
        
        roi = image[cy1:cy2, cx1:cx2]
        if roi.size == 0: return None
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_roi, axis=(0, 1))
        return np.array([avg_hsv[0]/180.0, avg_hsv[1]/255.0, avg_hsv[2]/255.0])

    def get_body_ratios(self, kps):
        nose = kps[0]
        ls, rs = kps[5], kps[6]
        le, lw = kps[7], kps[9]
        lh, rh = kps[11], kps[12]
        if ls[0] == 0 or rs[0] == 0: return None
        shoulder_w = self.calculate_distance(ls, rs)
        if shoulder_w < 10: return None
        features = np.array([-1.0, -1.0, -1.0])
        if lh[0] != 0 and rh[0] != 0:
            mid_s, mid_h = (ls + rs) / 2, (lh + rh) / 2
            torso_h = self.calculate_distance(mid_s, mid_h)
            if torso_h > 20:
                features[0] = shoulder_w / torso_h
                if le[0]!=0 and lw[0]!=0:
                    arm_len = self.calculate_distance(ls, le) + self.calculate_distance(le, lw)
                    features[1] = arm_len / torso_h
        if nose[0] != 0:
            mid_s = (ls + rs) / 2
            head_dist = self.calculate_distance(nose, mid_s)
            features[2] = head_dist / shoulder_w
        return features

    def calculate_total_score(self, target_data, candidate_data, img_w, img_h, time_lost):
        score = 0.0
        valid_parts = 0
        
        if time_lost > 0.5:
            w_spatial = 0.0 
            w_color = 0.8   
            w_body = 0.2
        else:
            w_spatial = self.weight_spatial_normal
            w_color = self.weight_color_normal
            w_body = self.weight_body_normal

        # --- 1. 颜色得分 (严打模式) ---
        color_diff = 0.0
        if target_data['color_hsv'] is not None and candidate_data['color_hsv'] is not None:
            t_hsv, c_hsv = target_data['color_hsv'], candidate_data['color_hsv']
            
            diff_h = abs(t_hsv[0] - c_hsv[0])
            if diff_h > 0.5: diff_h = 1.0 - diff_h
            
            diff_s = abs(t_hsv[1] - c_hsv[1])
            diff_v = abs(t_hsv[2] - c_hsv[2])
            avg_s = (t_hsv[1] + c_hsv[1]) / 2.0
            
            # 亮度否决 (0.2)
            if diff_v > 0.20: return 1.0 

            if avg_s < 0.2: color_diff = diff_v
            else: color_diff = (diff_h * 0.5) + (diff_s * 0.3) + (diff_v * 0.2)

            # 综合颜色否决 (0.2)
            if color_diff > 0.2: return 1.0 
            
            score += color_diff * w_color
            valid_parts += w_color

        # --- 2. 空间得分 ---
        if w_spatial > 0:
            if target_data['last_pos'] is not None and candidate_data['center'] is not None:
                dist = np.linalg.norm(target_data['last_pos'] - candidate_data['center'])
                max_dist = np.sqrt(img_w**2 + img_h**2)
                spatial_score = dist / max_dist
                if dist < 50: spatial_score = 0.0 
                score += spatial_score * w_spatial
                valid_parts += w_spatial

        # --- 3. 体型得分 ---
        if target_data['body_ratio'] is not None and candidate_data['fp'] is not None:
            diff_sum, cnt = 0.0, 0
            t_fp, c_fp = target_data['body_ratio'], candidate_data['fp']
            for i in range(3):
                if t_fp[i] > 0 and c_fp[i] > 0:
                    diff_sum += abs(t_fp[i] - c_fp[i])
                    cnt += 1
            if cnt > 0:
                score += (diff_sum / cnt) * w_body
                valid_parts += w_body

        return (score / valid_parts) if valid_parts > 0 else 999.0

    def rgb_callback(self, msg):
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            cv_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            if self.current_mode == 'RGB' and avg_brightness < self.brightness_threshold:
                self.current_mode = 'IR'
            elif self.current_mode == 'IR' and avg_brightness > (self.brightness_threshold + self.hysteresis):
                self.current_mode = 'RGB'

            if self.current_mode == 'IR':
                if self.latest_ir_frame is None: return
                final_image = cv2.cvtColor(self.latest_ir_frame, cv2.COLOR_GRAY2BGR)
                mode_txt = "IR (NIGHT)"
                color_ui = (0, 0, 255)
            else:
                final_image = cv_rgb
                mode_txt = "RGB (DAY)"
                color_ui = (0, 255, 0)

            final_image = cv2.resize(final_image, (640, 480))
            h, w = final_image.shape[:2]

            results = self.model.track(final_image, persist=True, conf=0.3, verbose=False, imgsz=320)
            
            detections = {}
            current_target_present = False
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
                
                for i, tid in enumerate(track_ids):
                    kps = keypoints[i] if i < len(keypoints) else []
                    fp = self.get_body_ratios(kps) if len(kps) > 0 else None
                    color_feat = self.get_color_features(final_image, boxes[i])
                    cx, cy = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                    
                    detections[tid] = {
                        'box': boxes[i], 'kps': kps, 'fp': fp, 
                        'center': np.array([cx, cy]), 'color_hsv': color_feat
                    }

            # --- 核心逻辑 ---
            
            # 1. 初始锁定 (C位优先)
            if self.target_id is None and not self.has_locked_once:
                best_id = None
                max_priority = -1.0 
                img_cx, img_cy = w / 2, h / 2

                for tid, data in detections.items():
                    width = data['box'][2] - data['box'][0]
                    height = data['box'][3] - data['box'][1]
                    area = width * height
                    
                    obj_cx, obj_cy = data['center']
                    dist_to_center = np.sqrt((obj_cx - img_cx)**2 + (obj_cy - img_cy)**2)
                    
                    priority = area / (dist_to_center + 100.0)
                    
                    if priority > max_priority:
                        max_priority = priority
                        best_id = tid
                        
                if best_id is not None:
                    self.target_id = best_id
                    self.display_id = best_id
                    self.has_locked_once = True
                    self.last_seen_time = current_time
                    self.color_sample_count = 0
                    self.color_locked = False
                    self.get_logger().info(f"FIRST LOCK: Real-ID {self.target_id}. (Center Priority)")

            # 2. 目标在场
            if self.target_id in detections:
                current_target_present = True
                self.last_seen_time = current_time 
                data = detections[self.target_id]
                self.target_features['last_pos'] = data['center']
                
                if data['fp'] is not None:
                    if self.target_features['body_ratio'] is None:
                        self.target_features['body_ratio'] = data['fp']
                    else:
                        old, new = self.target_features['body_ratio'], data['fp']
                        mask = new > 0
                        old[mask] = (1-self.update_rate)*old[mask] + self.update_rate*new[mask]
                
                # [宽松版学习逻辑]
                if data['color_hsv'] is not None and not self.color_locked:
                    cx, cy = data['center']
                    
                    # 只要中心点在中间区域即可 (宽容度高)
                    safe_x_min, safe_x_max = w * 0.25, w * 0.75
                    safe_y_min, safe_y_max = h * 0.1, h * 0.9
                    
                    is_centered = (safe_x_min < cx < safe_x_max) and (safe_y_min < cy < safe_y_max)
                    
                    if is_centered:
                        self.color_sample_count += 1
                        if self.target_features['color_hsv'] is None:
                            self.target_features['color_hsv'] = data['color_hsv']
                        else:
                            self.target_features['color_hsv'] = (1-self.update_rate)*self.target_features['color_hsv'] + \
                                                                self.update_rate*data['color_hsv']
                        
                        if self.color_sample_count >= self.frames_to_lock:
                            self.color_locked = True
                            self.get_logger().info(">>> COLOR LOCKED. SAFE TO MOVE. <<<")
                    else:
                        pass # 没在中间，暂停学习

            # 3. 目标丢失 -> Re-ID
            elif self.has_locked_once: 
                best_match_id = None
                min_score = 1.0 
                time_lost = current_time - self.last_seen_time
                
                for tid, data in detections.items():
                    score = self.calculate_total_score(self.target_features, data, w, h, time_lost)
                    self.get_logger().info(f"Checking ID {tid}: Score {score:.3f}")

                    if score < min_score:
                        min_score = score
                        best_match_id = tid
                
                if best_match_id is not None and min_score < 0.15:
                    old_real_id = self.target_id
                    self.target_id = best_match_id 
                    current_target_present = True
                    self.last_seen_time = current_time 
                    self.get_logger().warn(f"Re-ID Success: {old_real_id}->{self.target_id} (Score: {min_score:.3f})")
                else:
                    if best_match_id is not None:
                        self.get_logger().info(f"Stranger {best_match_id} REJECTED. Score {min_score:.3f}")

            # --- 绘图 ---
            for tid, data in detections.items():
                x1, y1, x2, y2 = map(int, data['box'])
                
                if tid == self.target_id:
                    color = (0, 0, 255)
                    label = f"TARGET [{self.display_id}]"
                    if self.color_locked: label += " [LOCKED]"
                    else:
                        progress = int((self.color_sample_count / self.frames_to_lock) * 100)
                        label += f" [LEARNING {progress}%]" 
                    
                    if self.target_features['color_hsv'] is not None:
                        hsv = self.target_features['color_hsv'].copy()
                        hsv[0]*=180; hsv[1]*=255; hsv[2]*=255
                        bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
                        cv2.circle(final_image, (x1+15, y1-15), 10, (int(bgr[0]), int(bgr[1]), int(bgr[2])), -1)
                else:
                    color = (0, 255, 0)
                    label = f"ID:{tid}"
                
                cv2.rectangle(final_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(final_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                kps = data['kps']
                if len(kps)>0:
                    ls, rs = kps[5], kps[6]
                    if ls[0]!=0 and rs[0]!=0:
                        cv2.line(final_image, (int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1])), (0,255,255), 2)

            status = f"{mode_txt} | Locked:{self.display_id}"
            if not current_target_present and self.has_locked_once: 
                time_lost = current_time - self.last_seen_time
                status += f" (SEARCHING... {time_lost:.1f}s)"
            cv2.putText(final_image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ui, 2)
            
            self.pub_annotated.publish(self.bridge.cv2_to_imgmsg(final_image, encoding="bgr8"))
            
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AutoSwitchTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()