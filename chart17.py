# robot_mimic_full_singlefile_fixed.py
"""
Robot Mimic - Single-file control panel + tracking
Focused on: accurate tracking, servo caps enforced, elbow mapping fixed,
no STT/Whisper, minimal AI stubs removed, single ready-to-run file.

Requirements:
pip install PySide6 opencv-python mediapipe numpy pyserial pyttsx3

Run:
python robot_mimic_full_singlefile_fixed.py
"""
import sys, os, time, threading, math, traceback, queue
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

# UI
from PySide6 import QtCore, QtGui, QtWidgets

# CV / tracking
import cv2
import numpy as np
import mediapipe as mp

# Serial (optional)
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except Exception:
    SERIAL_AVAILABLE = False

# TTS voice guidance (optional)
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# ---- Defaults / constants
DEFAULT_SERIAL_PORT = "COM3" if os.name == "nt" else "/dev/ttyUSB0"
DEFAULT_BAUD = 9600

DEFAULT_CAM_INDEX = 0
DEFAULT_CAM_W = 640
DEFAULT_CAM_H = 480
DEFAULT_FPS_CAP = 30

DEFAULT_SERVO_COUNT = 10
STEP_INCREMENT_DEG = 10
STEP_DELAY_SEC = 0.06

# Color BGR for OpenCV
COLOR_BONE = (0, 255, 0)
COLOR_LANDMARK = (0, 0, 255)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------- data classes
@dataclass
class ServoConfig:
    name: str = "servo"
    min_angle: int = 0
    max_angle: int = 180
    neutral: int = 90
    enabled: bool = True

@dataclass
class TrackerConfig:
    fps_cap: int = DEFAULT_FPS_CAP
    send_rate: int = 12
    smoothing_alpha: float = 0.45
    buffer_size: int = 5
    deadzone_deg: float = 0.5
    sensitivity: float = 1.0
    serial_port: str = DEFAULT_SERIAL_PORT
    baudrate: int = DEFAULT_BAUD
    voice_guidance: bool = True
    skip_calibration: bool = False
    step_sending: bool = True
    step_increment: int = STEP_INCREMENT_DEG
    step_delay: float = STEP_DELAY_SEC
    calibrate_hold_s: float = 1.6

# ---------------- helpers
def clamp(x, a, b): return max(a, min(b, x))
def angle_between_2d(a, b, c):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float); c = np.array(c, dtype=float)
    v1 = a - b; v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosang = np.dot(v1, v2) / denom
    cosang = clamp(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

# ---------------- SerialManager (safe)
class SerialManager:
    def __init__(self, monitor_callback=None):
        self.ser = None
        self.port = ""
        self.baud = DEFAULT_BAUD
        self.lock = threading.Lock()
        self.monitor_callback = monitor_callback
        self._step_thread = None
        self._stop_steps = threading.Event()
        self._last_sent = None

    @staticmethod
    def list_ports():
        if not SERIAL_AVAILABLE:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def _log(self, s):
        print("[SERIAL]", s)
        if self.monitor_callback:
            try: self.monitor_callback(s)
            except Exception: pass

    def open(self, port, baud):
        if not SERIAL_AVAILABLE:
            self._log("pyserial not installed")
            return False
        with self.lock:
            try:
                self.ser = serial.Serial(port, baud, timeout=0.02)
                time.sleep(0.05)
                self.port = port; self.baud = baud
                self._log(f"Opened {port}@{baud}")
                return True
            except Exception as e:
                self._log(f"Open failed: {e}")
                self.ser = None
                return False

    def close(self):
        with self.lock:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            if self._step_thread and self._step_thread.is_alive():
                self._stop_steps.set()
                self._step_thread.join(timeout=0.5)
                self._stop_steps.clear()

    def _write_line(self, line: str) -> bool:
        with self.lock:
            if not self.ser or not getattr(self.ser, "is_open", False):
                self._log("TX failed: not connected")
                return False
            try:
                self.ser.write(line.encode("utf-8"))
                try: self.ser.flush()
                except Exception: pass
                self._last_sent = line.strip()
                self._log("TX >> " + line.strip())
                return True
            except Exception as e:
                self._log("Write failed: " + str(e))
                try: self.ser.close()
                except Exception: pass
                self.ser = None
                return False

    def send_angles(self, angles: List[int], stepped=False, step_deg=STEP_INCREMENT_DEG, step_delay=STEP_DELAY_SEC):
        # Ensure final angles never exceed servo caps (clamping here is last line of defense)
        clamped = [int(clamp(round(a), 0, 180)) for a in angles]
        if not stepped:
            line = "S," + ",".join(map(str, clamped)) + "\n"
            return self._write_line(line)

        # stepped send
        if self._step_thread and self._step_thread.is_alive():
            self._stop_steps.set()
            self._step_thread.join(timeout=0.3)
            self._stop_steps.clear()

        def stepper(targets):
            cur = None
            try:
                if self._last_sent:
                    parts = self._last_sent.split(",")[1:]
                    cur = [int(p) for p in parts]
                else:
                    cur = [90] * len(targets)
            except Exception:
                cur = [90] * len(targets)
            sequences = []
            max_steps = 1
            for c, t in zip(cur, targets):
                if c == t:
                    seq = [c]
                else:
                    seq = []
                    if t > c:
                        v = c
                        while v < t:
                            v = min(t, v + step_deg); seq.append(int(v))
                    else:
                        v = c
                        while v > t:
                            v = max(t, v - step_deg); seq.append(int(v))
                sequences.append(seq); max_steps = max(max_steps, len(seq))
            # pad sequences with final values
            for i, seq in enumerate(sequences):
                if len(seq) < max_steps:
                    if seq:
                        sequences[i] = seq + [seq[-1]] * (max_steps - len(seq))
                    else:
                        sequences[i] = [cur[i]] * max_steps
            # send stepwise
            for sidx in range(max_steps):
                if self._stop_steps.is_set(): break
                pkt = "S," + ",".join(str(sequences[i][sidx]) for i in range(len(sequences))) + "\n"
                self._write_line(pkt)
                time.sleep(step_delay)
            return

        self._step_thread = threading.Thread(target=stepper, args=(clamped,), daemon=True)
        self._step_thread.start()
        return True

# ---------------- Camera / Tracking worker
class CameraWorker(threading.Thread):
    def __init__(self, cfg: TrackerConfig, servo_cfgs: List[ServoConfig],
                 frame_callback=None, telemetry_callback=None, serial_mgr: Optional[SerialManager]=None):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.servo_cfgs = servo_cfgs
        self.frame_callback = frame_callback
        self.telemetry_callback = telemetry_callback
        self.serial_mgr = serial_mgr

        # toggles
        self.toggle = {
            "mirror": True,
            "show_fps": True,
            "show_debug": False,
            "step_sending": cfg.step_sending,
            "buffer_smoothing": True,
            "skip_calibration": cfg.skip_calibration
        }
        self.landmarks = {"nose": True, "shoulders": True, "hands": True, "nose_neck_shoulder": True, "fingers": True}

        # visuals
        self.bone_color = COLOR_BONE
        self.landmark_color = COLOR_LANDMARK
        self.line_thickness = 2
        self.landmark_radius = 4
        self.hand_point_radius = 3

        # smoothing
        self.buffers = [deque(maxlen=max(1, self.cfg.buffer_size)) for _ in range(len(self.servo_cfgs))]
        self.smoothed = [sc.neutral for sc in self.servo_cfgs]

        # calibration
        self.calibrated = False
        self.calib_neutral = {"yaw": 0.0, "pitch": 0.0}
        self.calib_ranges = {"yaw": 20.0, "pitch": 20.0}
        self.calib_phase = {"active": False, "stage_text": "", "progress": 0.0}
        self.calib_recommend = {}

        # mediapipe
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=0,
                                 enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # camera
        self.cap = cv2.VideoCapture(DEFAULT_CAM_INDEX)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAM_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAM_H)
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps_cap)
        except Exception:
            pass

        self._stop = threading.Event()
        self._last_send_time = 0.0
        self._last_frame_time = time.time()
        self.fps = 0.0
        self.last_sent = [sc.neutral for sc in self.servo_cfgs]

    def stop(self):
        self._stop.set()

    def restart_camera(self, index, width, height):
        try: self.cap.release()
        except Exception: pass
        self.cap = cv2.VideoCapture(index)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps_cap)
        except Exception: pass

    # head estimation from nose/shoulders (unchanged)
    def estimate_head_yaw_pitch(self, pose_res, img_w, img_h):
        if not pose_res or not pose_res.pose_landmarks:
            return None, None
        try:
            lm = pose_res.pose_landmarks.landmark
            nose = lm[0]; left_sh = lm[11]; right_sh = lm[12]
            nx = nose.x * img_w; ny = nose.y * img_h
            lex = left_sh.x * img_w; rex = right_sh.x * img_w
            eye_mid_x = (lex + rex) / 2.0
            yaw = - ((nx - eye_mid_x) / (img_w / 4.0)) * 40.0
            shoulders_mid_y = ((left_sh.y + right_sh.y) * img_h) / 2.0
            pitch = - ((ny - shoulders_mid_y) / (img_h / 4.0)) * 30.0
            return float(yaw), float(pitch)
        except Exception:
            return None, None

    # hand curls approximations
    def compute_hand_curls(self, hand_lm):
        curls = {'index': 0.0, 'middle': 0.0, 'ring': 0.0, 'pinky': 0.0, 'thumb': 0.0}
        if not hand_lm:
            return curls
        try:
            lm = hand_lm.landmark
            def xy(i): return (lm[i].x, lm[i].y)
            fmap = {'index': (5,6,7), 'middle': (9,10,11), 'ring': (13,14,15), 'pinky': (17,18,19)}
            for name,(mcp,pip,dip) in fmap.items():
                a = xy(mcp); b = xy(pip); c = xy(dip)
                ang = angle_between_2d(a,b,c)
                lo, hi = 60.0, 180.0
                angc = clamp(ang, lo, hi)
                curl = (hi - angc) / (hi - lo)
                curls[name] = float(clamp(curl, 0.0, 1.0))
            # thumb
            try:
                a = xy(1); b = xy(2); c = xy(3)
                angt = angle_between_2d(a,b,c)
                lo_t, hi_t = 30.0, 160.0
                angt = clamp(angt, lo_t, hi_t)
                curls['thumb'] = float(clamp((hi_t - angt) / (hi_t - lo_t), 0.0, 1.0))
            except Exception:
                curls['thumb'] = 0.0
            return curls
        except Exception:
            return curls

    # ---------- KEY: compute servo targets (ELBOW FIX applied here)
    def compute_servo_targets(self, pose_res, hands_res, frame_shape):
        h, w = frame_shape
        targets = [sc.neutral for sc in self.servo_cfgs]

        # HEAD -> servo 0 (note: user requested neutral ~75)
        yaw_meas, pitch_meas = self.estimate_head_yaw_pitch(pose_res, w, h)
        neutral_yaw = self.calib_neutral['yaw'] if self.calibrated else 0.0
        obs_yaw = max(1e-6, self.calib_ranges.get('yaw', 20.0))
        span = 60.0 * self.cfg.sensitivity
        yaw_frac = 0.0 if yaw_meas is None else clamp((yaw_meas - neutral_yaw) / obs_yaw, -1.0, 1.0)
        # Map yaw fraction into servo span around neutral. ensure within servo caps later
        targets[0] = self.servo_cfgs[0].neutral + yaw_frac * (span / 2.0)

        # LEFT ARM mapping (servo 1..4)
        if pose_res and pose_res.pose_landmarks:
            try:
                lm = pose_res.pose_landmarks.landmark
                L_sh = lm[11]; L_el = lm[13]; L_wr = lm[15]
                # shoulder pitch (1)
                L_sh_y = L_sh.y * h; L_el_y = L_el.y * h
                l_pitch_frac = clamp((L_sh_y - L_el_y) / (h / 4.0), -1.0, 1.0)
                targets[1] = self.servo_cfgs[1].neutral + l_pitch_frac * (span / 2.0)
                # shoulder yaw (2)
                L_sh_x = L_sh.x * w; L_el_x = L_el.x * w
                l_yaw_frac = clamp((L_el_x - L_sh_x) / (w / 4.0), -1.0, 1.0)
                targets[2] = self.servo_cfgs[2].neutral + l_yaw_frac * (span / 2.0)
                # elbow (3) -- FIX: user wants servo angle INCREASE when elbow MOVES TOWARDS SHOULDER (i.e., when anatomical angle DECREASES)
                L_sh_pt = (L_sh.x * w, L_sh.y * h); L_el_pt = (L_el.x * w, L_el.y * h); L_wr_pt = (L_wr.x * w, L_wr.y * h)
                l_el_ang = angle_between_2d(L_sh_pt, L_el_pt, L_wr_pt)  # ~180 extended, ~30 flexed
                # Map so that smaller anatomical angle (flex) -> larger servo value (increasing)
                # i.e., invert mapping: anatomical 30 -> servo max, anatomical 180 -> servo min
                targets[3] = np.interp(l_el_ang, [30.0, 180.0], [self.servo_cfgs[3].max_angle, self.servo_cfgs[3].min_angle])
                # pronation approx (4)
                pron_frac = clamp((L_wr.x - L_el.x) * 4.0, -1.0, 1.0)
                targets[4] = self.servo_cfgs[4].neutral + pron_frac * (span / 4.0)
            except Exception:
                pass

        # RIGHT ARM mapping (servo 5..7)
        if pose_res and pose_res.pose_landmarks:
            try:
                lm = pose_res.pose_landmarks.landmark
                R_sh = lm[12]; R_el = lm[14]; R_wr = lm[16]
                R_sh_y = R_sh.y * h; R_el_y = R_el.y * h
                r_pitch_frac = clamp((R_sh_y - R_el_y) / (h / 4.0), -1.0, 1.0)
                targets[5] = self.servo_cfgs[5].neutral + r_pitch_frac * (span / 2.0)
                R_sh_x = R_sh.x * w; R_el_x = R_el.x * w
                r_yaw_frac = clamp((R_el_x - R_sh_x) / (w / 4.0), -1.0, 1.0)
                targets[6] = self.servo_cfgs[6].neutral + r_yaw_frac * (span / 2.0)
                R_sh_pt = (R_sh.x * w, R_sh.y * h); R_el_pt = (R_el.x * w, R_el.y * h); R_wr_pt = (R_wr.x * w, R_wr.y * h)
                r_el_ang = angle_between_2d(R_sh_pt, R_el_pt, R_wr_pt)
                # Invert mapping for right elbow as well so robot elbow extends with your extension
                targets[7] = np.interp(r_el_ang, [30.0, 180.0], [self.servo_cfgs[7].max_angle, self.servo_cfgs[7].min_angle])
            except Exception:
                pass

        # HANDS -> fingers (servo 8 combined 4 fingers; servo 9 thumb)
        try:
            if hands_res and getattr(hands_res, "multi_hand_landmarks", None):
                chosen = None
                if getattr(hands_res, "multi_handedness", None):
                    for i, hl in enumerate(hands_res.multi_hand_landmarks):
                        try:
                            label = hands_res.multi_handedness[i].classification[0].label
                            if label.lower().startswith("right"): chosen = hl; break
                        except Exception:
                            pass
                if chosen is None:
                    chosen = hands_res.multi_hand_landmarks[0]
                curls = self.compute_hand_curls(chosen)
                four_avg = float(np.mean([curls['index'], curls['middle'], curls['ring'], curls['pinky']]))
                targets[8] = np.interp(four_avg, [0.0, 1.0], [self.servo_cfgs[8].min_angle, self.servo_cfgs[8].max_angle])
                targets[9] = np.interp(curls['thumb'], [0.0, 1.0], [self.servo_cfgs[9].min_angle, self.servo_cfgs[9].max_angle])
        except Exception:
            pass

        # CLAMP to servo min/max (ultimate safety)
        final = []
        for i, t in enumerate(targets):
            try:
                v = float(t)
            except Exception:
                v = float(self.servo_cfgs[i].neutral)
            v = clamp(v, self.servo_cfgs[i].min_angle, self.servo_cfgs[i].max_angle)
            final.append(v)
        return final

    # draw overlays: nose->neck->shoulder included
    def draw_overlays(self, img, pose_res, hands_res):
        H, W = img.shape[:2]
        lc = self.landmark_color; bc = self.bone_color; lt = self.line_thickness
        def dot(x,y,r=3,color=(0,0,255)): cv2.circle(img, (int(x), int(y)), int(r), color, -1)
        try:
            if pose_res and pose_res.pose_landmarks:
                lm = pose_res.pose_landmarks.landmark
                ids = [11,12,13,14,15,16]
                pts = {}
                for i in ids:
                    if i < len(lm):
                        p = lm[i]; pts[i] = (int(p.x * W), int(p.y * H))
                        if self.landmarks.get("shoulders", True):
                            dot(pts[i][0], pts[i][1], r=self.landmark_radius, color=lc)
                if 11 in pts and 13 in pts: cv2.line(img, pts[11], pts[13], bc, lt)
                if 13 in pts and 15 in pts: cv2.line(img, pts[13], pts[15], bc, lt)
                if 12 in pts and 14 in pts: cv2.line(img, pts[12], pts[14], bc, lt)
                if 14 in pts and 16 in pts: cv2.line(img, pts[14], pts[16], bc, lt)
                if 11 in pts and 12 in pts: cv2.line(img, pts[11], pts[12], bc, lt)
                # nose to neck line: neck = midpoint of shoulders with slight offset for neck position
                try:
                    nose = pose_res.pose_landmarks.landmark[0]; nx, ny = int(nose.x * W), int(nose.y * H)
                    if self.landmarks.get("nose_neck_shoulder", True):
                        if 11 in pts and 12 in pts:
                            neck_x = int((pts[11][0] + pts[12][0]) / 2.0)
                            neck_y = int((pts[11][1] + pts[12][1]) / 2.0) + 8
                            cv2.line(img, (nx, ny), (neck_x, neck_y), bc, max(1, lt-1))
                            cv2.line(img, (neck_x, neck_y), pts[11], bc, max(1, lt-1))
                            cv2.line(img, (neck_x, neck_y), pts[12], bc, max(1, lt-1))
                    if self.landmarks.get("nose", True):
                        dot(nx, ny, r=self.landmark_radius, color=lc)
                except Exception:
                    pass
        except Exception:
            pass

        # hands
        try:
            if hands_res and getattr(hands_res, "multi_hand_landmarks", None) and self.landmarks.get("hands", True):
                for hand_lm in hands_res.multi_hand_landmarks:
                    for idx, lm in enumerate(hand_lm.landmark):
                        x = int(lm.x * W); y = int(lm.y * H)
                        if self.landmarks.get("fingers", True):
                            if idx in (4,8,12,16,20):
                                dot(x, y, r=self.hand_point_radius+1, color=lc)
                            else:
                                dot(x, y, r=max(1, self.hand_point_radius-1), color=lc)
                    mp_drawing.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=lc, thickness=max(1, lt-1), circle_radius=self.hand_point_radius),
                                              mp_drawing.DrawingSpec(color=bc, thickness=lt))
                    try:
                        wp = hand_lm.landmark[0]; wppt = (int(wp.x * W), int(wp.y * H))
                        for base in (5,9,13,17):
                            b = hand_lm.landmark[base]; bp = (int(b.x*W), int(b.y*H))
                            cv2.line(img, wppt, bp, bc, max(1, lt-1))
                    except Exception: pass
        except Exception:
            pass

        # fps
        if self.toggle.get("show_fps", True):
            try:
                cv2.putText(img, f"FPS: {int(self.fps)}", (W-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            except Exception: pass

    # run loop
    def run(self):
        frame_interval = 1.0 / max(1, self.cfg.fps_cap)
        while not self._stop.is_set():
            t0 = time.time()
            ret, raw = self.cap.read()
            if not ret:
                time.sleep(0.01); continue
            frame = cv2.flip(raw, 1) if self.toggle.get("mirror", True) else raw.copy()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = hands_res = None
            try:
                pose_res = self.pose.process(rgb)
                hands_res = self.hands.process(rgb)
            except Exception:
                pass

            targets = self.compute_servo_targets(pose_res, hands_res, (h,w))
            now = time.time()
            send_interval = 1.0 / max(1, self.cfg.send_rate)
            if now - self._last_send_time >= send_interval:
                smoothed_list = []
                alpha = clamp(self.cfg.smoothing_alpha, 0.0, 1.0)
                for i, tgt in enumerate(targets):
                    buf = self.buffers[i]
                    buf.append(tgt)
                    avg = float(np.mean(buf)) if len(buf) else float(tgt)
                    if self.toggle.get("buffer_smoothing", True):
                        new = self.smoothed[i] * (1.0 - alpha) + avg * alpha
                    else:
                        new = avg
                    neu = self.servo_cfgs[i].neutral
                    if abs(new - neu) < self.cfg.deadzone_deg:
                        new = neu
                    # final clamp to ensure never exceed servo caps
                    new = clamp(new, self.servo_cfgs[i].min_angle, self.servo_cfgs[i].max_angle)
                    self.smoothed[i] = new
                    smoothed_list.append(int(round(new)))
                # send to serial
                ok = False
                if self.serial_mgr:
                    stepped = bool(self.toggle.get("step_sending", True))
                    ok = self.serial_mgr.send_angles(smoothed_list, stepped=stepped, step_deg=self.cfg.step_increment, step_delay=self.cfg.step_delay)
                self.last_sent = smoothed_list.copy()
                self._last_send_time = now
                if self.telemetry_callback:
                    try:
                        self.telemetry_callback({"servo_values": self.last_sent, "serial_ok": bool(self.serial_mgr and self.serial_mgr.ser), "fps": int(self.fps)})
                    except Exception:
                        pass

            # draw overlays and top text
            annotated = frame.copy()
            try: self.draw_overlays(annotated, pose_res, hands_res)
            except Exception: pass
            try:
                txt = " | ".join(f"{i}:{v}" for i,v in enumerate(self.last_sent))
                cv2.rectangle(annotated, (0,0), (annotated.shape[1], 28), (0,0,0), -1)
                cv2.putText(annotated, txt, (6,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            except Exception: pass

            t1 = time.time(); self.fps = 1.0 / max(1e-6, t1 - self._last_frame_time); self._last_frame_time = t1

            if self.frame_callback:
                try: self.frame_callback(annotated)
                except Exception: pass

            elapsed = time.time() - t0
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0: time.sleep(sleep_time)

        # cleanup
        try: self.cap.release()
        except Exception: pass
        try: self.pose.close()
        except Exception: pass
        try: self.hands.close()
        except Exception: pass

# ----------------- UI / MainWindow (PySide6)
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Mimic Control Panel - Fixed (No Whisper)")
        self.resize(1200, 820)

        # config & servos
        self.cfg = TrackerConfig()
        self.servo_cfgs = self._create_default_servo_configs()

        # managers
        self.serial_mgr = SerialManager(monitor_callback=self.append_serial_monitor)
        self.voice_engine = None
        if VOICE_AVAILABLE:
            try:
                self.voice_engine = pyttsx3.init()
                self.voice_engine.setProperty('rate', 160)
            except Exception:
                self.voice_engine = None

        # camera worker
        self.camera_worker = CameraWorker(self.cfg, self.servo_cfgs, frame_callback=self.receive_frame, telemetry_callback=self.receive_telemetry, serial_mgr=self.serial_mgr)
        self.camera_worker.start()

        # UI build
        self._build_ui()

        # frame queue & display timer
        self.frame_queue = queue.Queue(maxsize=2)
        self.display_timer = QtCore.QTimer(); self.display_timer.timeout.connect(self._display_frame); self.display_timer.start(30)

        # greeting
        self._play_voice("WELCOME ABOARD CAPTAIN")

    def _create_default_servo_configs(self):
        names = [
            "Head (s1 yaw only)",                   # servo 0
            "L Shoulder Pitch (s2) - MEAN=120",    # 1
            "L Shoulder Yaw (s3)",                 # 2
            "L Elbow (s4) - MAX 110 MIN 0",        # 3
            "L Pronation (s5) - MEAN=90",          # 4
            "R Shoulder Pitch",                    # 5
            "R Shoulder Yaw",                      # 6
            "R Elbow",                             # 7
            "Fingers (4 combined)",                # 8
            "Thumb"                                # 9
        ]
        cfgs = []
        for i in range(DEFAULT_SERVO_COUNT):
            if i == 0:      # HEAD neutral updated to 75 per your instruction
                cfgs.append(ServoConfig(name=names[i], min_angle=0, max_angle=150, neutral=75, enabled=True))
            elif i == 1:
                cfgs.append(ServoConfig(name=names[i], min_angle=60, max_angle=180, neutral=120, enabled=True))
            elif i == 3:
                cfgs.append(ServoConfig(name=names[i], min_angle=0, max_angle=110, neutral=60, enabled=True))
            else:
                cfgs.append(ServoConfig(name=names[i], min_angle=0, max_angle=180, neutral=90, enabled=True))
        return cfgs

    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        vmain = QtWidgets.QVBoxLayout(central)

        # ribbon bar
        ribbon = QtWidgets.QHBoxLayout(); vmain.addLayout(ribbon)
        self._add_ribbon_group(ribbon, "Home", [("Greet", self.on_greet), ("Theme", self.on_toggle_theme)])
        self._add_ribbon_group(ribbon, "Tracking", [("Start", self.on_start_tracking), ("Stop", self.on_stop_tracking), ("Mirror", self.on_toggle_mirror), ("Step Send", self.on_toggle_step_send)])
        self._add_ribbon_group(ribbon, "Calibration", [("Start Directed", self.on_start_calibration), ("Skip Calib", self.on_skip_calib), ("Calib Time...", self.on_set_calib_time)])
        self._add_ribbon_group(ribbon, "Appearance", [("Landmark Color", self.on_pick_landmark_color), ("Bone Color", self.on_pick_bone_color), ("Line Width", self.on_set_line_width)])
        self._add_ribbon_group(ribbon, "Servos", [("Servo Settings", self.on_open_servo_settings), ("Show/Hide Bottom Angles", self.on_toggle_bottom_angles)])
        self._add_ribbon_group(ribbon, "Serial", [("Refresh Ports", self.on_refresh_ports), ("Connect", self.on_connect_serial), ("Disconnect", self.on_disconnect_serial), ("Serial Monitor", self.on_open_serial_monitor)])
        self._add_ribbon_group(ribbon, "Help", [("Keyboard Help", self.on_show_help), ("About", self.on_show_about)])

        main_h = QtWidgets.QHBoxLayout(); vmain.addLayout(main_h)
        left_col = QtWidgets.QVBoxLayout(); main_h.addLayout(left_col, 3)

        # Sensitivity group
        sens_grp = QtWidgets.QGroupBox("Sensitivity & Performance"); left_col.addWidget(sens_grp)
        sens_layout = QtWidgets.QGridLayout(sens_grp)
        sens_layout.addWidget(QtWidgets.QLabel("Sensitivity"), 0, 0)
        self.sens_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sens_slider.setRange(10, 400)
        self.sens_slider.setValue(int(self.cfg.sensitivity * 100)); self.sens_slider.valueChanged.connect(self.on_sens_changed)
        sens_layout.addWidget(self.sens_slider, 0, 1); self.sens_label = QtWidgets.QLabel(f"{self.cfg.sensitivity:.2f}"); sens_layout.addWidget(self.sens_label, 0, 2)
        sens_layout.addWidget(QtWidgets.QLabel("Smoothing α"), 1, 0)
        self.smooth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(int(self.cfg.smoothing_alpha * 100)); self.smooth_slider.valueChanged.connect(self.on_smooth_changed)
        sens_layout.addWidget(self.smooth_slider, 1, 1); self.smooth_label = QtWidgets.QLabel(f"{self.cfg.smoothing_alpha:.2f}"); sens_layout.addWidget(self.smooth_label, 1, 2)
        sens_layout.addWidget(QtWidgets.QLabel("Buffer size"), 2, 0)
        self.buf_spin = QtWidgets.QSpinBox(); self.buf_spin.setRange(1, 30); self.buf_spin.setValue(self.cfg.buffer_size); self.buf_spin.valueChanged.connect(self.on_buf_changed)
        sens_layout.addWidget(self.buf_spin, 2, 1)
        sens_layout.addWidget(QtWidgets.QLabel("Camera FPS cap"), 3, 0)
        self.fps_spin = QtWidgets.QSpinBox(); self.fps_spin.setRange(5, 60); self.fps_spin.setValue(self.cfg.fps_cap); self.fps_spin.valueChanged.connect(self.on_fps_changed)
        sens_layout.addWidget(self.fps_spin, 3, 1)
        sens_layout.addWidget(QtWidgets.QLabel("Send rate (Hz)"), 4, 0)
        self.send_spin = QtWidgets.QSpinBox(); self.send_spin.setRange(1, 60); self.send_spin.setValue(self.cfg.send_rate); self.send_spin.valueChanged.connect(self.on_send_rate_changed)
        sens_layout.addWidget(self.send_spin, 4, 1)

        # calibration group (with (i) info)
        calib_grp = QtWidgets.QGroupBox("Calibration"); left_col.addWidget(calib_grp)
        cal_layout = QtWidgets.QHBoxLayout(calib_grp)
        self.calib_status_label = QtWidgets.QLabel("Not calibrated"); cal_layout.addWidget(self.calib_status_label)
        self.start_calib_btn = QtWidgets.QPushButton("Start Directed Calibration"); self.start_calib_btn.clicked.connect(self.on_start_calibration)
        cal_layout.addWidget(self.start_calib_btn)
        self.skip_calib_chk = QtWidgets.QCheckBox("Skip calibration"); self.skip_calib_chk.setChecked(self.cfg.skip_calibration); self.skip_calib_chk.stateChanged.connect(self.on_skip_calib_changed)
        cal_layout.addWidget(self.skip_calib_chk)
        info_btn = QtWidgets.QToolButton(); info_btn.setText("i"); info_btn.clicked.connect(lambda: self._show_info_dialog("Skip Calibration", "Skipping calibration will use current neutral values. Only enable if you know what you're doing."))
        cal_layout.addWidget(info_btn)

        # camera controls
        cam_grp = QtWidgets.QGroupBox("Camera"); left_col.addWidget(cam_grp)
        cam_layout = QtWidgets.QGridLayout(cam_grp)
        cam_layout.addWidget(QtWidgets.QLabel("Index"), 0, 0)
        self.cam_index_spin = QtWidgets.QSpinBox(); self.cam_index_spin.setRange(0, 5); self.cam_index_spin.setValue(DEFAULT_CAM_INDEX); cam_layout.addWidget(self.cam_index_spin, 0, 1)
        cam_layout.addWidget(QtWidgets.QLabel("Width"), 0, 2); self.cam_w_spin = QtWidgets.QSpinBox(); self.cam_w_spin.setRange(160,1920); self.cam_w_spin.setValue(DEFAULT_CAM_W); cam_layout.addWidget(self.cam_w_spin, 0, 3)
        cam_layout.addWidget(QtWidgets.QLabel("Height"), 0, 4); self.cam_h_spin = QtWidgets.QSpinBox(); self.cam_h_spin.setRange(120,1080); self.cam_h_spin.setValue(DEFAULT_CAM_H); cam_layout.addWidget(self.cam_h_spin,0,5)
        self.restart_cam_btn = QtWidgets.QPushButton("Restart Camera"); self.restart_cam_btn.clicked.connect(self.on_restart_camera); cam_layout.addWidget(self.restart_cam_btn, 0, 6)

        # bottom: servo angles always visible (with tooltip for min/max/neutral)
        bottom_servo_widget = QtWidgets.QWidget(); bottom_layout = QtWidgets.QHBoxLayout(bottom_servo_widget)
        self.bottom_servo_labels = []
        for i, sc in enumerate(self.servo_cfgs):
            lbl = QtWidgets.QLabel(f"{i}: --")
            lbl.setStyleSheet("color: #CFCFCF; padding:2px;")
            bottom_layout.addWidget(lbl)
            self.bottom_servo_labels.append(lbl)
            # initial tooltip
            lbl.setToolTip(f"{sc.name}\nMin: {sc.min_angle}\nNeutral: {sc.neutral}\nMax: {sc.max_angle}")
        left_col.addWidget(bottom_servo_widget)

        # right column: serial + ai stub
        right_col = QtWidgets.QVBoxLayout(); main_h.addLayout(right_col, 1)
        serial_box = QtWidgets.QGroupBox("Serial Monitor"); right_col.addWidget(serial_box)
        s_layout = QtWidgets.QVBoxLayout(serial_box)
        self.serial_text = QtWidgets.QTextEdit(); self.serial_text.setReadOnly(True); s_layout.addWidget(self.serial_text)
        btn_row = QtWidgets.QHBoxLayout(); s_layout.addLayout(btn_row)
        self.btn_refresh_ports = QtWidgets.QPushButton("Refresh Ports"); self.btn_refresh_ports.clicked.connect(self.on_refresh_ports); btn_row.addWidget(self.btn_refresh_ports)
        self.btn_connect_serial = QtWidgets.QPushButton("Connect"); self.btn_connect_serial.clicked.connect(self.on_connect_serial); btn_row.addWidget(self.btn_connect_serial)
        self.btn_disconnect_serial = QtWidgets.QPushButton("Disconnect"); self.btn_disconnect_serial.clicked.connect(self.on_disconnect_serial); btn_row.addWidget(self.btn_disconnect_serial)

        # voice controls
        voice_box = QtWidgets.QGroupBox("Voice Guidance (TTS only)"); right_col.addWidget(voice_box)
        v_layout = QtWidgets.QHBoxLayout(voice_box)
        self.voice_chk = QtWidgets.QCheckBox("Voice guidance"); self.voice_chk.setChecked(self.cfg.voice_guidance); v_layout.addWidget(self.voice_chk)
        self.voice_test_btn = QtWidgets.QPushButton("Voice Test"); self.voice_test_btn.clicked.connect(lambda: self._play_voice("Voice test, one two three")); v_layout.addWidget(self.voice_test_btn)

        # footer
        footer = QtWidgets.QHBoxLayout(); vmain.addLayout(footer)
        self.status_bar_label = QtWidgets.QLabel("Status: Idle | Serial: Disconnected"); footer.addWidget(self.status_bar_label)
        quit_btn = QtWidgets.QPushButton("Quit"); quit_btn.clicked.connect(self.on_quit); footer.addWidget(quit_btn)

        # dark theme
        self._apply_dark_theme()

        # open separate camera window (OpenCV)
        cv2.namedWindow("Camera - Tracking", cv2.WINDOW_NORMAL)

    def _add_ribbon_group(self, layout, title, buttons):
        grp = QtWidgets.QGroupBox(title); h = QtWidgets.QHBoxLayout(grp)
        for (label, cb) in buttons:
            btn = QtWidgets.QPushButton(label); btn.clicked.connect(cb); h.addWidget(btn)
        layout.addWidget(grp)

    # ----- actions / callbacks
    def on_greet(self): self._play_voice("Welcome aboard Captain.")
    def on_toggle_theme(self):
        idx = getattr(self, "_theme_idx", 0); idx = (idx + 1) % 3; self._theme_idx = idx
        if idx == 0: self._apply_dark_theme()
        elif idx == 1: self._apply_amoled_theme()
        else: self._apply_light_theme()
        self._play_voice("Theme switched")

    def on_start_tracking(self):
        if not self.camera_worker.is_alive():
            self.camera_worker = CameraWorker(self.cfg, self.servo_cfgs, frame_callback=self.receive_frame, telemetry_callback=self.receive_telemetry, serial_mgr=self.serial_mgr)
            self.camera_worker.start()
        self.status_bar_label.setText("Status: Tracking started")

    def on_stop_tracking(self):
        try: self.camera_worker.stop(); self.status_bar_label.setText("Status: Tracking stopped")
        except Exception: pass

    def on_toggle_mirror(self):
        v = not self.camera_worker.toggle.get("mirror", True); self.camera_worker.toggle["mirror"] = v; self.status_bar_label.setText(f"Mirror = {v}")

    def on_toggle_step_send(self):
        v = not self.camera_worker.toggle.get("step_sending", True); self.camera_worker.toggle["step_sending"] = v; self.status_bar_label.setText(f"Step sending = {v}")

    def on_start_calibration(self):
        if self.camera_worker.toggle.get("skip_calibration", False):
            QtWidgets.QMessageBox.information(self, "Calibration", "Skip calibration is enabled"); return
        threading.Thread(target=self._calibration_routine, daemon=True).start()

    def on_skip_calib(self):
        v = not self.camera_worker.toggle.get("skip_calibration", False); self.camera_worker.toggle["skip_calibration"] = v
        self.skip_calib_chk.setChecked(v); self.status_bar_label.setText(f"Skip calibration = {v}")

    def on_skip_calib_changed(self, state):
        v = (state == QtCore.Qt.CheckState.Checked); self.camera_worker.toggle["skip_calibration"] = v; self.status_bar_label.setText(f"Skip calibration set to {v}")

    def on_set_calib_time(self):
        val, ok = QtWidgets.QInputDialog.getDouble(self, "Calibration hold time", "Seconds per stage:", value=self.cfg.calibrate_hold_s, min=0.4, max=10.0, decimals=2)
        if ok:
            self.cfg.calibrate_hold_s = float(val)
            self._play_voice(f"Calibration hold time set to {val} seconds")

    def on_pick_landmark_color(self):
        col = QtWidgets.QColorDialog.getColor(initial=QtGui.QColor(*COLOR_LANDMARK[::-1]))
        if col.isValid():
            r,g,b = col.red(), col.green(), col.blue(); self.camera_worker.landmark_color = (b,g,r)

    def on_pick_bone_color(self):
        col = QtWidgets.QColorDialog.getColor(initial=QtGui.QColor(*COLOR_BONE[::-1]))
        if col.isValid():
            r,g,b = col.red(), col.green(), col.blue(); self.camera_worker.bone_color = (b,g,r)

    def on_set_line_width(self):
        val, ok = QtWidgets.QInputDialog.getInt(self, "Line thickness", "Enter 1..10:", value=self.camera_worker.line_thickness, min=1, max=10)
        if ok: self.camera_worker.line_thickness = int(val)

    def on_open_servo_settings(self):
        dlg = ServoSettingsDialog(self, self.servo_cfgs)
        if dlg.exec_():
            # update tooltips & ensure bottom labels reflect min/max/neutral
            for i, sc in enumerate(self.servo_cfgs):
                if i < len(self.bottom_servo_labels):
                    self.bottom_servo_labels[i].setToolTip(f"{sc.name}\nMin: {sc.min_angle}\nNeutral: {sc.neutral}\nMax: {sc.max_angle}")

    def on_toggle_bottom_angles(self):
        for lbl in self.bottom_servo_labels: lbl.setVisible(not lbl.isVisible())

    def on_refresh_ports(self):
        ports = SerialManager.list_ports()
        if not ports:
            QtWidgets.QMessageBox.information(self, "Ports", "No serial ports found")
        else:
            QtWidgets.QMessageBox.information(self, "Ports", "Found: " + ", ".join(ports))

    def on_connect_serial(self):
        if not SERIAL_AVAILABLE:
            QtWidgets.QMessageBox.critical(self, "Serial", "pyserial not installed"); return
        port, ok = QtWidgets.QInputDialog.getText(self, "Serial port", "Enter port (e.g. COM3):", text=self.cfg.serial_port)
        if not ok or not port: return
        baud, ok = QtWidgets.QInputDialog.getInt(self, "Baud", "Enter baudrate:", value=self.cfg.baudrate)
        if not ok: return
        ok2 = self.serial_mgr.open(port, baud)
        if ok2:
            self.status_bar_label.setText(f"Serial: Connected {port}@{baud}")
            self.cfg.serial_port = port; self.cfg.baudrate = baud
        else:
            QtWidgets.QMessageBox.critical(self, "Serial", "Open failed")

    def on_disconnect_serial(self):
        self.serial_mgr.close(); self.status_bar_label.setText("Serial: Disconnected")

    def on_open_serial_monitor(self):
        self.serial_text.raise_()

    def on_show_help(self):
        s = (
            "Keyboard controls:\n"
            "Q/ESC - Quit\n"
            "C - Recalibrate neutral position\n"
            "K - Toggle skip calibration on restart\n"
            "F - Freeze tracking\n"
            "M - Toggle mirror\n"
            "I - Toggle inverted (flip up/down)\n"
            "D - Toggle debug overlay\n"
            "V - Toggle FPS display\n"
            "S - Save current frame as image\n"
            "P - Print current statistics\n"
            "H - Show this help\n"
        )
        QtWidgets.QMessageBox.information(self, "Keyboard Help", s)

    def on_show_about(self):
        QtWidgets.QMessageBox.information(self, "About", "Robot Mimic - focused tracking & servo safety")

    def on_sens_changed(self, v):
        s = max(0.05, float(v) / 100.0)
        self.cfg.sensitivity = s; self.sens_label.setText(f"{s:.2f}")

    def on_smooth_changed(self, v):
        a = clamp(float(v) / 100.0, 0.0, 1.0); self.cfg.smoothing_alpha = a; self.smooth_label.setText(f"{a:.2f}")

    def on_buf_changed(self):
        v = int(self.buf_spin.value()); self.cfg.buffer_size = v
        self.camera_worker.buffers = [deque(maxlen=v) for _ in range(len(self.servo_cfgs))]
        self.status_bar_label.setText(f"Buffer size set to {v}")

    def on_fps_changed(self, v):
        v = int(self.fps_spin.value()); self.cfg.fps_cap = v; self.camera_worker.cfg.fps_cap = v

    def on_send_rate_changed(self, v):
        v = int(self.send_spin.value()); self.cfg.send_rate = v; self.camera_worker.cfg.send_rate = v

    def on_restart_camera(self):
        idx = int(self.cam_index_spin.value()); w = int(self.cam_w_spin.value()); h = int(self.cam_h_spin.value())
        self.camera_worker.restart_camera(idx, w, h); self.status_bar_label.setText(f"Camera restarted {idx} {w}x{h}")

    def on_ask_ai(self):
        # AI stub removed — no heavy model included. Provide fallback messages.
        self._play_voice("AI offline. This is a tracking-first build.")

    def append_serial_monitor(self, s):
        try:
            self.serial_text.append(s)
            self.serial_text.verticalScrollBar().setValue(self.serial_text.verticalScrollBar().maximum())
        except Exception:
            pass

    def receive_frame(self, frame):
        try:
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        except Exception:
            pass

    def _display_frame(self):
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                if getattr(self.camera_worker, "calib_phase", None) and self.camera_worker.calib_phase.get("active", False):
                    txt = self.camera_worker.calib_phase.get("stage_text", ""); prog = self.camera_worker.calib_phase.get("progress", 0.0)
                    self._draw_calibration_overlay(frame, txt, prog)
                cv2.imshow("Camera - Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and key in (ord('q'), 27):
                self.on_quit()
        except Exception:
            pass

    def receive_telemetry(self, d):
        try:
            sv = d.get("servo_values", None)
            if sv:
                for i, v in enumerate(sv):
                    if i < len(self.bottom_servo_labels):
                        self.bottom_servo_labels[i].setText(f"{i}: {v}")
            serial_ok = d.get("serial_ok", False); fps = d.get("fps", None)
            s = f"Serial: {'OK' if serial_ok else 'Disconnected'}"
            if fps is not None: s += f" | FPS: {fps}"
            self.status_bar_label.setText(s)
        except Exception:
            pass

    def _calibration_routine(self):
        hold = getattr(self.cfg, 'calibrate_hold_s', 1.6)
        steps = [("Keep your head straight and look at the camera", hold),
                 ("Turn your head LEFT as far as comfortable", hold),
                 ("Turn your head RIGHT as far as comfortable", hold),
                 ("Look UP as far as comfortable", hold),
                 ("Look DOWN as far as comfortable", hold)]
        self.camera_worker.calib_phase = {"active": True, "stage_text": "", "progress": 0.0}
        self._play_voice("Starting calibration. Follow instructions.")
        yaw_samples = []; pitch_samples = []
        max_left = None; max_right = None; max_up = None; max_down = None
        try:
            for i, (txt, dur) in enumerate(steps):
                self.camera_worker.calib_phase["stage_text"] = txt
                if self.voice_engine and self.voice_chk.isChecked():
                    self._play_voice(txt)
                endt = time.time() + dur
                local_y = []; local_p = []
                while time.time() < endt:
                    try:
                        local_y.append(float(self.camera_worker.smoothed[0]))
                        if len(self.camera_worker.smoothed) > 1:
                            local_p.append(float(self.camera_worker.smoothed[1]))
                    except Exception:
                        pass
                    self.camera_worker.calib_phase["progress"] = 1.0 - (endt - time.time()) / dur
                    time.sleep(0.04)
                if i == 0:
                    yaw_samples.extend(local_y); pitch_samples.extend(local_p)
                elif i == 1 and local_y:
                    max_left = min(local_y) if max_left is None else min(max_left, min(local_y))
                elif i == 2 and local_y:
                    max_right = max(local_y) if max_right is None else max(max_right, max(local_y))
                elif i == 3 and local_p:
                    max_up = max(local_p) if max_up is None else max(max_up, max(local_p))
                elif i == 4 and local_p:
                    max_down = min(local_p) if max_down is None else min(max_down, min(local_p))
                self.camera_worker.calib_phase["progress"] = 0.0
                time.sleep(0.08)
        finally:
            self.camera_worker.calib_phase = {"active": False, "stage_text": "", "progress": 0.0}
        derived_neutral_y = float(np.mean(yaw_samples)) if yaw_samples else 0.0
        derived_neutral_p = float(np.mean(pitch_samples)) if pitch_samples else 0.0
        left_delta = abs(derived_neutral_y - max_left) if max_left is not None else 20.0
        right_delta = abs(max_right - derived_neutral_y) if max_right is not None else 20.0
        up_delta = abs(max_up - derived_neutral_p) if max_up is not None else 20.0
        down_delta = abs(derived_neutral_p - max_down) if max_down is not None else 20.0
        obs_yaw = max(left_delta, right_delta, 10.0); obs_pitch = max(up_delta, down_delta, 10.0)
        self.camera_worker.calib_neutral = {"yaw": derived_neutral_y, "pitch": derived_neutral_p}
        self.camera_worker.calib_ranges = {"yaw": obs_yaw, "pitch": obs_pitch}
        self.camera_worker.calibrated = True
        span = 60.0 * self.cfg.sensitivity
        rmin0 = int(clamp(self.servo_cfgs[0].neutral - span / 2.0, 0, 180)); rmax0 = int(clamp(self.servo_cfgs[0].neutral + span / 2.0, 0, 180))
        rmin1 = int(clamp(self.servo_cfgs[1].neutral - span / 2.0, 0, 180)); rmax1 = int(clamp(self.servo_cfgs[1].neutral + span / 2.0, 0, 180))
        self.camera_worker.calib_recommend = {0: {"min": rmin0, "max": rmax0, "neutral": self.servo_cfgs[0].neutral},
                                             1: {"min": rmin1, "max": rmax1, "neutral": self.servo_cfgs[1].neutral}}
        self.calib_status_label.setText("Calibrated")
        self._play_voice("Calibration complete. Recommendations populated.")
        self.status_bar_label.setText("Calibration complete")

    def _draw_calibration_overlay(self, frame, text, progress=None):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 100), (w, h // 2 + 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 1.0; thickness = 2
        words = text.split(); lines = []; curr = []
        for word in words:
            test = " ".join(curr + [word])
            size = cv2.getTextSize(test, font, font_scale, thickness)[0]
            if size[0] < w - 100:
                curr.append(word)
            else:
                if curr: lines.append(" ".join(curr)); curr = [word]
        if curr: lines.append(" ".join(curr))
        y_offset = h // 2 - (len(lines) * 30) // 2
        for line in lines:
            size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (w - size[0]) // 2
            cv2.putText(frame, line, (x, y_offset), font, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
            cv2.putText(frame, line, (x, y_offset), font, font_scale, (0,255,255), thickness, cv2.LINE_AA)
            y_offset += 36
        if progress is not None:
            bar_w = 400; bar_h = 18; bx = (w - bar_w) // 2; by = h // 2 + 60
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (80,80,80), -1)
            fill = int(bar_w * clamp(progress, 0.0, 1.0))
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), (0,255,255), -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (255,255,255), 2)
            pct = f"{int(clamp(progress, 0.0, 1.0) * 100)}%"
            cv2.putText(frame, pct, (bx + bar_w + 8, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    def _play_voice(self, text: str):
        if not VOICE_AVAILABLE or not self.voice_engine:
            print("[VOICE]", text); return
        def speak(t):
            try:
                self.voice_engine.say(t); self.voice_engine.runAndWait()
            except Exception as e:
                print("Voice error:", e)
        threading.Thread(target=speak, args=(text,), daemon=True).start()

    def _show_info_dialog(self, title, text):
        QtWidgets.QMessageBox.information(self, title, text)

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #1E1E1E; color: #DDD; }
            QGroupBox { color: #DDD; border: 1px solid #333; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px 0 3px; }
            QPushButton { background: #2D2D30; color: #EEE; border: 1px solid #3A3A3A; padding: 6px; }
            QLabel { color: #D0D0D0; }
            QTextEdit { background: #0F0F10; color: #DDD; }
            QLineEdit { background: #0F0F10; color: #DDD; }
            QSlider::groove:horizontal { background: #333; height: 8px; }
            QSlider::handle:horizontal { background: #AAA; width: 14px; }
        """)

    def _apply_amoled_theme(self):
        self.setStyleSheet("""
            QMainWindow { background: #000; color: #EEE; }
            QGroupBox { color: #EEE; border: 1px solid #222; }
            QPushButton { background: #111; color: #FFF; border: 1px solid #333; padding: 6px; }
            QLabel { color: #EEE; }
        """)

    def _apply_light_theme(self):
        self.setStyleSheet("")

    def on_quit(self):
        try: self.camera_worker.stop()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        try: self.serial_mgr.close()
        except Exception: pass
        QtWidgets.QApplication.quit()

# ---------------- Servo Settings Dialog (clear labels + enforce caps)
class ServoSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent, servo_cfgs: List[ServoConfig]):
        super().__init__(parent)
        self.setWindowTitle("Servo Settings - Min / Neutral / Max (angles enforced)")
        self.servo_cfgs = servo_cfgs
        self.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(self)
        canvas = QtWidgets.QWidget(); layout.addWidget(canvas)
        gl = QtWidgets.QGridLayout(canvas)
        # header labels
        gl.addWidget(QtWidgets.QLabel("Index"), 0, 0)
        gl.addWidget(QtWidgets.QLabel("Name"), 0, 1)
        gl.addWidget(QtWidgets.QLabel("Min"), 0, 2)
        gl.addWidget(QtWidgets.QLabel("Max"), 0, 3)
        gl.addWidget(QtWidgets.QLabel("Neutral"), 0, 4)
        gl.addWidget(QtWidgets.QLabel("Enabled"), 0, 5)
        self.widgets = []
        for i, sc in enumerate(self.servo_cfgs):
            gl.addWidget(QtWidgets.QLabel(str(i)), i+1, 0)
            name = QtWidgets.QLineEdit(sc.name); gl.addWidget(name, i+1, 1)
            min_sp = QtWidgets.QSpinBox(); min_sp.setRange(0, 180); min_sp.setValue(sc.min_angle); gl.addWidget(min_sp, i+1, 2)
            max_sp = QtWidgets.QSpinBox(); max_sp.setRange(0, 180); max_sp.setValue(sc.max_angle); gl.addWidget(max_sp, i+1, 3)
            neu_sp = QtWidgets.QSpinBox(); neu_sp.setRange(0, 180); neu_sp.setValue(sc.neutral); gl.addWidget(neu_sp, i+1, 4)
            en_chk = QtWidgets.QCheckBox(); en_chk.setChecked(sc.enabled); gl.addWidget(en_chk, i+1, 5)
            self.widgets.append((name, min_sp, max_sp, neu_sp, en_chk))
        btn_row = QtWidgets.QHBoxLayout(); layout.addLayout(btn_row)
        btn_apply = QtWidgets.QPushButton("Apply"); btn_apply.clicked.connect(self.on_apply); btn_row.addWidget(btn_apply)
        btn_close = QtWidgets.QPushButton("Close"); btn_close.clicked.connect(self.close); btn_row.addWidget(btn_close)

    def on_apply(self):
        # Apply and enforce min <= neutral <= max
        for i, (name, min_sp, max_sp, neu_sp, en_chk) in enumerate(self.widgets):
            try:
                nm = name.text(); mi = int(min_sp.value()); ma = int(max_sp.value()); neu = int(neu_sp.value()); ena = bool(en_chk.isChecked())
                # enforce min <= neutral <= max; if invalid swap or clamp
                if mi > ma:
                    mi, ma = ma, mi
                neu = clamp(neu, mi, ma)
                self.servo_cfgs[i].name = nm if nm.strip() else self.servo_cfgs[i].name
                self.servo_cfgs[i].min_angle = int(mi); self.servo_cfgs[i].max_angle = int(ma); self.servo_cfgs[i].neutral = int(neu); self.servo_cfgs[i].enabled = ena
            except Exception:
                pass
        QtWidgets.QMessageBox.information(self, "Servo Settings", "Applied. Min <= Neutral <= Max enforced.")

# ---------------- Main entrypoint
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error please restart the entire projevt  :", e)
        traceback.print_exc()
