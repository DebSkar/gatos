import os
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
"""
Webcam gesture -> meme detector (desktop version).

Abre dos ventanas lado a lado:
  - "Camera": video de la webcam con landmarks y HUD en vivo
  - "Meme": imagen del meme de gatito que coincide con el gesto actual

Gestos soportados (definidos en Gestos.txt):
  1. orejaGato      -> 1.Oreja de gato.jpeg (Dos manos arriba de la cabeza, dedos indice y medio estirados hacia abajo)
  2. mirarAbajo     -> 2.MirarHaciaAbajo.jpeg (Cabeza hacia abajo / Pitch negativo)
  3. lengua         -> 3.Lengua.jpeg (Sacar la lengua / apertura bucal)
  4. cabezaAtras    -> 4.CabezaHaciaAtras.jpeg (Inclinar cabeza hacia atrás / Pitch positivo)
  5. brazosAtras    -> 5.Brazos hacia atras.jpeg (Levantar brazos y echar hacia atrás/lados)
  6. shocked        -> 6.Shocked.jpeg (Cruzar brazos a altura del pecho)
  7. duda           -> 7.Duda.jpeg (Cara duda, indice estirado tocando el lado de la cabeza)
  8. pistola        -> 8.Pistola.jpeg (Dedos pistola / Indice y medio juntos arriba y pulgar al lado)
  - default         -> pokercat.jpg (Estado neutro / sin gesto)

Presiona 'q' o ESC para salir.
"""

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "Gatitoactualizado"

GESTURE_MEMES = {
    "default": ["7.Duda.jpeg"],
    "orejaGato": ["1.Oreja de gato.jpeg"],
    "mirarAbajo": ["2.MirarHaciaAbajo.jpeg"],
    "lengua": ["3.Lengua.jpeg"],
    "cabezaAtras": ["4.CabezaHaciaAtras.jpeg"],
    "brazosAtras": ["5.Brazos hacia atras.jpeg"],
    "shocked": ["6.Shocked.jpeg"],
    "duda": ["7.Duda.jpeg"],
    "pistola": ["8.Pistola.jpeg"],
}

GESTURE_NAMES = {
    "default": "En reposo (Haz un gesto)",
    "orejaGato": "1. Orejas de Gato",
    "mirarAbajo": "2. Mirar Hacia Abajo",
    "lengua": "3. Sacar la Lengua",
    "cabezaAtras": "4. Cabeza Hacia Atras",
    "brazosAtras": "5. Brazos Hacia Atras",
    "shocked": "6. Brazos Cruzados (Shocked)",
    "duda": "7. Cara de Duda",
    "pistola": "8. Pistola",
}

# Gestos cuyos memes son videos (si aplica)
VIDEO_GESTURES = set()

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- Geometry Helpers ----------------------------------------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def head_pose_angles(matrix):
    """Extrae Yaw (giro izq/der) y Pitch (inclinación arriba/abajo) en grados."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        yaw = 0.0
        pitch = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
    else:
        yaw = math.degrees(math.atan2(-r[2, 0], sy))
        pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "middleTip": pts[12],
        "ringTip": pts[16],
        "pinkyTip": pts[20],
        "thumbTip": pts[4],
        "indexMCP": pts[5],
        "middleMCP": pts[9],
        "wrist": pts[0],
        "palmCenter": pts[9],
        "pts": pts,
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


class GestureState:
    def __init__(self):
        self.last_face = None
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.last_pitch_debug = 0.0
        self.last_mouth_open_debug = 0.0

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            nose_tip, forehead, chin = p3(f[1]), p3(f[10]), p3(f[152])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek) or 1e-6
            mouth_open = dist(upper_lip, lower_lip) / face_width

            nose_forehead_dist = abs(nose_tip[1] - forehead[1]) + 1e-6
            chin_nose_dist = abs(chin[1] - nose_tip[1])
            vertical_ratio = chin_nose_dist / nose_forehead_dist

            yaw_deg, pitch_deg = 0.0, 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg, pitch_deg = head_pose_angles(face_result.facial_transformation_matrixes[0])

            self.last_face = {
                "mouthCenter": mouth_center,
                "faceWidth": face_width,
                "mouthOpen": mouth_open,
                "yawDeg": yaw_deg,
                "pitchDeg": pitch_deg,
                "verticalRatio": vertical_ratio,
                "rightCheek": right_cheek,
                "leftCheek": left_cheek,
                "t": now,
            }
            self.last_yaw_debug = yaw_deg
            self.last_pitch_debug = pitch_deg
            self.last_mouth_open_debug = mouth_open
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and (now - self.last_face["t"]) < FACE_STALE_MS
        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks] if hand_result.hand_landmarks else []

        # --- 1. GESTOS DE 2 MANOS ---
        if len(hands) >= 2:
            # Gesto 1: orejaGato (Dos manos arriba de la cabeza, índice y medio extendidos)
            both_ears = all(
                h["indexUp"] and h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]
                for h in hands[:2]
            )
            if both_ears:
                if face_is_fresh:
                    head_top = self.last_face["mouthCenter"][1] - self.last_face["faceWidth"] * 0.6
                    both_high = all(h["palmCenter"][1] < head_top or h["wrist"][1] < head_top for h in hands[:2])
                else:
                    both_high = all(h["palmCenter"][1] < 0.45 for h in hands[:2])
                if both_high:
                    return "orejaGato"

            # Gesto 5: brazosAtras (Levantar brazos y separar hacia los costados / atrás)
            hands_dist_x = abs(hands[0]["wrist"][0] - hands[1]["wrist"][0])
            both_elevated = all(h["wrist"][1] < 0.75 for h in hands[:2])
            if both_elevated and hands_dist_x > 0.45:
                return "brazosAtras"

            # Gesto 6: shocked (Cruzar brazos a la altura del pecho)
            if face_is_fresh:
                chest_y = self.last_face["mouthCenter"][1] + self.last_face["faceWidth"] * 0.2
                both_chest = all(h["wrist"][1] > chest_y for h in hands[:2])
            else:
                both_chest = all(h["wrist"][1] > 0.45 for h in hands[:2])
            
            wrists_close_x = abs(hands[0]["wrist"][0] - hands[1]["wrist"][0]) < 0.35
            wrists_close_y = abs(hands[0]["wrist"][1] - hands[1]["wrist"][1]) < 0.30
            if both_chest and wrists_close_x and wrists_close_y:
                return "shocked"

        # --- 2. GESTOS DE 1 MANO (o evaluados por mano individual) ---
        for h in hands:
            # Gesto 7: duda (Cara de duda, índice estirado tocando el lado de la cabeza/sien)
            if is_pointing(h) and face_is_fresh:
                face = self.last_face
                d_r = dist(h["indexTip"], face["rightCheek"]) / face["faceWidth"]
                d_l = dist(h["indexTip"], face["leftCheek"]) / face["faceWidth"]
                near_temple = (d_r < 0.55 or d_l < 0.55) and (h["indexTip"][1] < face["mouthCenter"][1] - face["faceWidth"] * 0.15)
                if near_temple:
                    return "duda"

            # Gesto 8: pistola (Dedos índice y medio juntos hacia arriba, pulgar hacia el lado)
            is_gun = (
                h["indexUp"] and h["middleUp"] and not h["ringUp"] and not h["pinkyUp"] and h["thumbOut"]
            )
            if is_gun:
                tip_gap = dist(h["indexTip"], h["middleTip"]) / h["handScale"]
                if tip_gap < 0.50:
                    return "pistola"

        # --- 3. GESTOS FACIALES (sin manos o prioritarios en rostro) ---
        if face_is_fresh:
            face = self.last_face

            # Gesto 3: lengua (Sacar la lengua / apertura bucal)
            if face["mouthOpen"] > 0.24:
                return "lengua"

            # Gesto 4: cabezaAtras (Inclinar cabeza hacia atrás / mirar hacia arriba)
            if face["pitchDeg"] > 13.0 or face["verticalRatio"] > 1.55:
                return "cabezaAtras"

            # Gesto 2: mirarAbajo (Cabeza hacia abajo / mirar hacia el suelo)
            if face["pitchDeg"] < -13.0 or face["verticalRatio"] < 0.45:
                return "mirarAbajo"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            continue
        imgs = []
        for name in files:
            p = MEMES / name
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {p}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    nombre_gesto = GESTURE_NAMES.get(gesture, gesture)
    lines = [
        f"--- GATITO ACTUALIZADO (8 GESTOS) ---",
        f"Gesto: {nombre_gesto}",
        f"Pitch: {state.last_pitch_debug:+.1f} deg  (atras >+13, abajo <-13)",
        f"Mouth Open: {state.last_mouth_open_debug:.2f}  (lengua >0.24)",
        f"Yaw: {state.last_yaw_debug:+.1f} deg",
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def open_camera():
    """Intenta abrir la cámara web probando índices (0, 1, 2) y backend AVFoundation en macOS."""
    print("Iniciando búsqueda de cámara web...")
    for idx in [0, 1, 2]:
        # 1. Intentar con AVFoundation (nativo macOS)
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ok, test_frame = cap.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    print(f"-> Cámara abierta exitosamente en índice {idx} (backend AVFoundation).")
                    return cap
                cap.release()
        except Exception:
            pass

        # 2. Intentar backend por defecto
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, test_frame = cap.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    print(f"-> Cámara abierta exitosamente en índice {idx}.")
                    return cap
                cap.release()
        except Exception:
            pass

    return None


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS / "hand_landmarker.task"),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(MODELS / "face_landmarker.task"),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()

    cap = open_camera()
    if cap is None:
        print("\n" + "=" * 70)
        print("AVISO / ERROR DE CÁMARA EN MACOS:")
        print("No se pudo acceder a la cámara web en los índices 0, 1 ni 2.")
        print("Causas frecuentes:")
        print("1. Falta de permisos en macOS:")
        print("   Ve a: Configuración del Sistema -> Privacidad y Seguridad -> Cámara")
        print("   y activa el interruptor para Terminal, iTerm o tu editor de código.")
        print("2. La cámara está siendo utilizada exclusivamente por otra aplicación")
        print("   (Zoom, FaceTime, Meet, navegador web, etc.). Ciérrala y reintenta.")
        print("=" * 70 + "\n")
        raise RuntimeError("No se pudo abrir la cámara web (permiso denegado o dispositivo no disponible)")

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                current_meme = random.choice(memes[gesture])

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()