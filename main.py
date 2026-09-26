import time
import numpy as np
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer

from hand_mapping import STRAIGHT_FRACS, fracs_from_landmarks, forearm_roll_frac, palm_side, ENABLE_FOREARM_ROLL

MODEL_PATH = "shadow_hand/scene_dual.xml"
SMOOTHING = 0.20

class HandActuatorIndex:
    def __init__(self, model, prefix):
        self.idx = {}
        self.lo = {}
        self.hi = {}

        for suffix in STRAIGHT_FRACS:
            name = f"{prefix}_A_{suffix}"
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"No existe {name}")
            lo, hi = model.actuator_ctrlrange[aid]
            self.idx[suffix] = aid
            self.lo[suffix] = lo
            self.hi[suffix] = hi

    def apply(self, ctrl, fracs, blend):
        for suffix, frac in fracs.items():
            i = self.idx[suffix]
            target = self.lo[suffix] + frac * (self.hi[suffix] - self.lo[suffix])
            ctrl[i] = blend * ctrl[i] + (1.0 - blend) * target

def main():
    print(f"Cargando modelo: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    hands_idx = {
        "rh": HandActuatorIndex(model, "rh"),
        "lh": HandActuatorIndex(model, "lh")
    }

    ctrl = np.zeros(model.nu)
    for hand in hands_idx.values():
        hand.apply(ctrl, STRAIGHT_FRACS, 0.0)
    data.ctrl[:] = ctrl

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "frontal")
        if cam_id != -1:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        print("Cámara iniciada. Pulsa Q para salir.")
        prev_time = time.time()

        while viewer.is_running() and cap.isOpened():
            ok, frame = cap.read()
            if not ok: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    lm = [(p.x, p.y, p.z) for p in hand_lm.landmark]
                    label = handedness.classification[0].label
                    prefix = "rh" if label == "Right" else "lh"
                    color = (0, 255, 0) if prefix == "rh" else (255, 128, 0)

                    mp_draw.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=color, thickness=2)
                    )

                    name = "Derecha -> mano derecha robot" if prefix == "rh" else "Izquierda -> mano izquierda robot"
                    y = 30 if prefix == "rh" else 60
                    cv2.putText(frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    fracs = fracs_from_landmarks(lm, prefix)
                    if ENABLE_FOREARM_ROLL:
                        fracs["FRJ"] = forearm_roll_frac(lm, prefix)

                    hands_idx[prefix].apply(ctrl, fracs, SMOOTHING)

            data.ctrl[:] = ctrl
            now = time.time()
            elapsed = now - prev_time
            prev_time = now

            steps = min(50, max(1, int(round(elapsed / model.opt.timestep))))
            for _ in range(steps):
                mujoco.mj_step(model, data)

            viewer.sync()
            cv2.imshow("Camara", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()