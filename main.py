import time
import numpy as np
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer

MODEL_PATH = "shadow_hand/scene_dual.xml"

# Índices de landmarks de MediaPipe Hands (21 puntos por mano)
WRIST = 0
THUMB = [1, 2, 3, 4]        # CMC, MCP, IP, TIP
INDEX = [5, 6, 7, 8]        # MCP, PIP, DIP, TIP
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
LITTLE = [17, 18, 19, 20]

# Suavizado (0 = sin suavizado, más cerca de 1 = más suave/lento)
SMOOTHING = 0.2

# Sufijos de actuador comunes a ambas manos (cada mano tiene rh_A_<suf> / lh_A_<suf>)
FINGER_ACTUATORS = {
    "FF": ["FFJ4", "FFJ3", "FFJ0"],
    "MF": ["MFJ4", "MFJ3", "MFJ0"],
    "RF": ["RFJ4", "RFJ3", "RFJ0"],
    "LF": ["LFJ5", "LFJ4", "LFJ3", "LFJ0"],
}


def angle_2d(a, b, c):
    """Ángulo en grados en el vértice b, formado por los puntos a-b-c (usa solo x,y)."""
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def curl_from_angle(deg, straight=175.0, bent=50.0):
    """Convierte un ángulo articular en un valor de curvatura 0 (recto) a 1 (doblado)."""
    val = (straight - deg) / (straight - bent)
    return float(np.clip(val, 0.0, 1.0))


def finger_curl(lm, chain):
    """Curvatura media de un dedo (0=abierto, 1=cerrado) a partir de sus 4 puntos."""
    mcp, pip, dip, tip = [lm[i] for i in chain]
    a1 = angle_2d(mcp, pip, dip)
    a2 = angle_2d(pip, dip, tip)
    return float(np.clip((curl_from_angle(a1) + curl_from_angle(a2)) / 2.0, 0.0, 1.0))


def thumb_curl(lm):
    cmc, mcp, ip_, tip = [lm[i] for i in THUMB]
    a1 = angle_2d(cmc, mcp, ip_)
    a2 = angle_2d(mcp, ip_, tip)
    return float(np.clip((curl_from_angle(a1, straight=170, bent=110) +
                           curl_from_angle(a2, straight=170, bent=110)) / 2.0, 0.0, 1.0))


def thumb_spread(lm):
    """Qué tan separado está el pulgar de la palma (0=pegado, 1=abierto)."""
    wrist = lm[WRIST]
    index_mcp = lm[INDEX[0]]
    thumb_tip = lm[THUMB[3]]
    palm_size = np.linalg.norm(np.array(index_mcp[:2]) - np.array(wrist[:2])) + 1e-6
    dist = np.linalg.norm(np.array(thumb_tip[:2]) - np.array(index_mcp[:2]))
    return float(np.clip(dist / (palm_size * 1.6), 0.0, 1.0))


def fracs_from_landmarks(lm):
    """A partir de los 21 landmarks de UNA mano (lista de (x, y, z)) calcula,
    para cada sufijo de actuador (ej. 'FFJ3', 'THJ1', ...), la fracción 0..1
    dentro de su ctrlrange. Es independiente de si la mano es derecha o
    izquierda: eso solo decide con qué prefijo (rh_/lh_) se aplica."""
    fracs = {"WRJ2": 0.5, "WRJ1": 0.5}

    for prefix, chain in (("FF", INDEX), ("MF", MIDDLE), ("RF", RING), ("LF", LITTLE)):
        c = finger_curl(lm, chain)
        fracs[f"{prefix}J4"] = 0.5   # separación lateral: neutral
        fracs[f"{prefix}J3"] = c      # flexión nudillo (MCP)
        fracs[f"{prefix}J0"] = c      # flexión PIP+DIP (tendón)

    lf_curl = finger_curl(lm, LITTLE)
    fracs["LFJ5"] = lf_curl * 0.4   # arco metacarpiano del meñique

    t_curl = thumb_curl(lm)
    t_spread = thumb_spread(lm)
    fracs["THJ5"] = 0.5
    fracs["THJ4"] = 1.0 - t_spread
    fracs["THJ3"] = 0.5
    fracs["THJ2"] = t_curl
    fracs["THJ1"] = t_curl
    return fracs


def straight_fracs():
    """Fracciones de la posición de reposo: mano recta y abierta."""
    fracs = {"WRJ2": 0.5, "WRJ1": 0.5}
    for prefix in ("FF", "MF", "RF"):
        fracs[f"{prefix}J4"] = 0.5
        fracs[f"{prefix}J3"] = 0.0
        fracs[f"{prefix}J0"] = 0.0
    fracs["LFJ4"] = 0.5
    fracs["LFJ3"] = 0.0
    fracs["LFJ0"] = 0.0
    fracs["LFJ5"] = 0.0
    fracs["THJ5"] = 0.5
    fracs["THJ4"] = 0.5
    fracs["THJ3"] = 0.5
    fracs["THJ2"] = 0.0
    fracs["THJ1"] = 0.0
    return fracs


STRAIGHT_FRACS = straight_fracs()


class HandActuatorIndex:
    """Resuelve, para un prefijo dado ('rh' o 'lh'), los índices y rangos
    de ctrl de cada actuador <prefix>_A_<sufijo>, una sola vez."""

    def __init__(self, model, prefix):
        self.prefix = prefix
        self.idx = {}
        self.lo = {}
        self.hi = {}
        for suf in STRAIGHT_FRACS.keys():
            name = f"{prefix}_A_{suf}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"No existe el actuador '{name}' en el modelo")
            lo, hi = model.actuator_ctrlrange[aid]
            self.idx[suf] = aid
            self.lo[suf] = lo
            self.hi[suf] = hi

    def apply(self, ctrl, fracs, blend):
        """Mezcla exponencialmente ctrl[hand] hacia los targets dados por
        `fracs`, con blend = peso del valor anterior (SMOOTHING)."""
        for suf, frac in fracs.items():
            i = self.idx[suf]
            target = self.lo[suf] + frac * (self.hi[suf] - self.lo[suf])
            ctrl[i] = blend * ctrl[i] + (1 - blend) * target


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    rh = HandActuatorIndex(model, "rh")
    lh = HandActuatorIndex(model, "lh")
    hands_idx = {"rh": rh, "lh": lh}

    ctrl = np.zeros(model.nu)
    rh.apply(ctrl, STRAIGHT_FRACS, blend=0.0)
    lh.apply(ctrl, STRAIGHT_FRACS, blend=0.0)
    data.ctrl[:] = ctrl

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (índice 0). Prueba con otro índice.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Abrir el visor ya mirando desde la cámara "frontal" definida en la
        # escena (de frente a las palmas, como la webcam), en vez de la
        # cámara libre por defecto.
        frontal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "frontal")
        if frontal_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = frontal_id

        print("Listo. Muestra una o las dos manos frente a la cámara. Pulsa 'q' para salir.")
        prev_time = time.time()
        # Tope de pasos de física por frame: evita que, si un frame tarda
        # mucho (ej. una pausa del sistema), intentemos "recuperar" tiempo
        # de golpe con miles de mj_step y se cuelgue la ventana.
        MAX_STEPS_PER_FRAME = 50
        while viewer.is_running() and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected = {"rh": False, "lh": False}

            if result.multi_hand_landmarks:
                for hand_lm, handedness in zip(result.multi_hand_landmarks,
                                                result.multi_handedness):
                    lm = [(p.x, p.y, p.z) for p in hand_lm.landmark]
                    raw_label = handedness.classification[0].label  # "Right" / "Left"
                    prefix = "rh" if raw_label == "Right" else "lh"
                    detected[prefix] = True

                    color = (0, 255, 0) if prefix == "rh" else (255, 128, 0)
                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=color, thickness=2),
                    )
                    label_text = "Derecha -> mano derecha robot" if prefix == "rh" \
                        else "Izquierda -> mano izquierda robot"
                    y = 30 if prefix == "rh" else 60
                    cv2.putText(frame, label_text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    fracs = fracs_from_landmarks(lm)
                    hands_idx[prefix].apply(ctrl, fracs, blend=SMOOTHING)

            # Cualquier mano no detectada este frame vuelve suavemente a reposo
            for prefix, was_seen in detected.items():
                if not was_seen:
                    hands_idx[prefix].apply(ctrl, STRAIGHT_FRACS, blend=SMOOTHING)

            data.ctrl[:] = ctrl

            # Avanzar la física lo que realmente pasó en tiempo real desde
            # el frame anterior, en vez de un único mj_step de 2 ms. Así el
            # reloj simulado no se queda atrás del reloj real y los
            # actuadores (que son un servo de posición, no un salto
            # instantáneo) tienen tiempo simulado suficiente para alcanzar
            # el target en cada frame.
            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            n_steps = min(MAX_STEPS_PER_FRAME,
                          max(1, int(round(elapsed / model.opt.timestep))))
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
            viewer.sync()

            cv2.imshow("Camara", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()