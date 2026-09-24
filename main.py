import time
import numpy as np
import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer

MODEL_PATH = "shadow_hand/scene_dual.xml"
SMOOTHING = 0.20

WRIST = 0
THUMB = [1, 2, 3, 4]
INDEX = [5, 6, 7, 8]
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
LITTLE = [17, 18, 19, 20]

FINGER_STRAIGHT_ANGLE = 175.0
FINGER_BENT_ANGLE = 50.0
SPREAD_ANGLE = 22.0
SPREAD_GAIN = 0.85
J4_MIN_FRAC = 0.20
J4_MAX_FRAC = 0.80

ENABLE_THUMB_ROTATION = True
THUMB_NEUTRAL_ANGLE = 75.0
THUMB_ANGLE_SPAN = 55.0

ENABLE_WRIST_FLEX = False
ENABLE_FOREARM_ROLL = True
FRJ_DEPTH_SCALE = 0.20

SPREAD_SIGN = {
    "rh": {
        "FF": 1.0,
        "MF": 1.0,
        "RF": -1.0,
        "LF": -1.0
    },
    "lh": {
        "FF": -1.0,
        "MF": -1.0,
        "RF": 1.0,
        "LF": 1.0
    },
}


def angle_2d(a, b, c):
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    return float(np.degrees(np.arccos(np.clip(
        np.dot(v1, v2) / (n1 * n2),
        -1.0,
        1.0
    ))))


def curl_from_angle(
    deg,
    straight=FINGER_STRAIGHT_ANGLE,
    bent=FINGER_BENT_ANGLE
):
    return float(np.clip(
        (straight - deg) / (straight - bent),
        0.0,
        1.0
    ))


def finger_curl(lm, chain):
    mcp, pip, dip, tip = [lm[i] for i in chain]

    c1 = curl_from_angle(
        angle_2d(mcp, pip, dip)
    )

    c2 = curl_from_angle(
        angle_2d(pip, dip, tip)
    )

    return float(np.clip(
        (c1 + c2) / 2.0,
        0.0,
        1.0
    ))


def finger_mcp_curl(lm, chain):
    mcp = np.array(lm[chain[0]][:2])
    pip = np.array(lm[chain[1]][:2])
    wrist = np.array(lm[WRIST][:2])

    a = mcp - wrist
    b = pip - mcp

    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    ang = np.degrees(np.arccos(np.clip(
        np.dot(a, b) / (n1 * n2),
        -1.0,
        1.0
    )))

    return float(np.clip(
        (ang - 10.0) / 100.0,
        0.0,
        1.0
    ))


def finger_lateral_angle(lm, chain):
    mcp = np.array(lm[chain[0]][:2])
    pip = np.array(lm[chain[1]][:2])

    v = pip - mcp

    return float(np.clip(
        np.degrees(np.arctan2(v[0], -v[1])),
        -SPREAD_ANGLE,
        SPREAD_ANGLE
    ))


def lateral_spread_frac(lm, chain, sign):
    angle = finger_lateral_angle(lm, chain)

    angle *= sign * SPREAD_GAIN

    return float(np.clip(
        0.5 + (angle / SPREAD_ANGLE) * 0.5,
        J4_MIN_FRAC,
        J4_MAX_FRAC
    ))


def thumb_curl(lm):
    cmc, mcp, ip, tip = [lm[i] for i in THUMB]

    c1 = curl_from_angle(
        angle_2d(cmc, mcp, ip),
        170.0,
        110.0
    )

    c2 = curl_from_angle(
        angle_2d(mcp, ip, tip),
        170.0,
        110.0
    )

    return float(np.clip(
        (c1 + c2) / 2.0,
        0.0,
        1.0
    ))


def thumb_spread(lm):
    wrist = np.array(lm[WRIST][:2])
    index_mcp = np.array(lm[INDEX[0]][:2])
    thumb_tip = np.array(lm[THUMB[3]][:2])

    palm = np.linalg.norm(
        index_mcp - wrist
    ) + 1e-6

    dist = np.linalg.norm(
        thumb_tip - index_mcp
    )

    return float(np.clip(
        dist / (palm * 1.6),
        0.0,
        1.0
    ))


def thumb_rotation_frac(lm):
    angle = angle_2d(
        lm[MIDDLE[0]],
        lm[WRIST],
        lm[THUMB[1]]
    )

    delta = np.clip(
        angle - THUMB_NEUTRAL_ANGLE,
        -THUMB_ANGLE_SPAN,
        THUMB_ANGLE_SPAN
    )

    return float(np.clip(
        0.5 + 0.5 * delta / THUMB_ANGLE_SPAN,
        0.0,
        1.0
    ))


def wrist_fracs(lm):
    wrist = np.array(lm[WRIST])

    avg = np.mean([
        lm[INDEX[0]],
        lm[MIDDLE[0]],
        lm[RING[0]],
        lm[LITTLE[0]]
    ], axis=0)

    v = avg - wrist

    angle = np.clip(
        np.degrees(np.arctan2(v[0], -v[1])),
        -40.0,
        40.0
    )

    wrj2 = 0.5 + (angle / 40.0) * 0.5

    dz = np.clip(
        (avg[2] - wrist[2]) / 0.15,
        -1.0,
        1.0
    )

    wrj1 = (
        0.5 + dz * 0.5
        if ENABLE_WRIST_FLEX
        else 0.5
    )

    return {
        "WRJ2": float(np.clip(wrj2, 0.0, 1.0)),
        "WRJ1": float(np.clip(wrj1, 0.0, 1.0))
    }


def forearm_roll_frac(lm, hand_prefix):
    wrist = np.array(lm[WRIST], dtype=float)
    middle = np.array(lm[MIDDLE[0]], dtype=float)
    index = np.array(lm[INDEX[0]], dtype=float)
    pinky = np.array(lm[LITTLE[0]], dtype=float)

    axis = middle - wrist
    n = np.linalg.norm(axis)

    if n < 1e-8:
        return 0.5

    axis /= n

    if hand_prefix == "rh":
        normal = np.cross(pinky - wrist, index - wrist)
    else:
        normal = np.cross(index - wrist, pinky - wrist)

    n = np.linalg.norm(normal)

    if n < 1e-8:
        return 0.5

    normal /= n

    camera = np.array([
        0.0,
        0.0,
        -1.0
    ])

    reference = (
        camera
        - np.dot(camera, axis) * axis
    )

    n = np.linalg.norm(reference)

    if n < 1e-8:
        fallback = np.array([
            1.0,
            0.0,
            0.0
        ])

        reference = (
            fallback
            - np.dot(fallback, axis) * axis
        )

        n = np.linalg.norm(reference)

        if n < 1e-8:
            return 0.5

    reference /= n

    normal_proj = (
        normal
        - np.dot(normal, axis) * axis
    )

    n = np.linalg.norm(normal_proj)

    if n < 1e-8:
        return 0.5

    normal_proj /= n

    angle = np.arctan2(
        np.dot(
            axis,
            np.cross(
                reference,
                normal_proj
            )
        ),
        np.dot(
            reference,
            normal_proj
        )
    )

    angle *= -1.0

    return float(np.clip(
        0.5 + angle / (2.0 * np.pi),
        0.0,
        1.0
    ))


def palm_side(lm, hand_prefix):
    wrist = np.array(lm[WRIST], dtype=float)
    index = np.array(lm[INDEX[0]], dtype=float)
    pinky = np.array(lm[LITTLE[0]], dtype=float)

    # Corrección de quiralidad para la palma
    if hand_prefix == "rh":
        normal = np.cross(pinky - wrist, index - wrist)
    else:
        normal = np.cross(index - wrist, pinky - wrist)

    n = np.linalg.norm(normal)

    if n < 1e-8:
        return "LADO"

    normal /= n

    facing = np.dot(
        normal,
        [0.0, 0.0, -1.0]
    )

    if facing > 0.20:
        return "PALMA"

    if facing < -0.20:
        return "DORSO"

    return "LADO"


def fracs_from_landmarks(lm, hand_prefix):
    fracs = {}

    fracs.update(
        wrist_fracs(lm)
    )

    fingers = [
        ("FF", INDEX),
        ("MF", MIDDLE),
        ("RF", RING),
        ("LF", LITTLE)
    ]

    for prefix, chain in fingers:
        curl = finger_curl(
            lm,
            chain
        )

        mcp = finger_mcp_curl(
            lm,
            chain
        )

        fracs[f"{prefix}J4"] = (
            lateral_spread_frac(
                lm,
                chain,
                SPREAD_SIGN[hand_prefix][prefix]
            )
        )

        fracs[f"{prefix}J3"] = float(
            np.clip(
                0.65 * mcp + 0.35 * curl,
                0.0,
                1.0
            )
        )

        fracs[f"{prefix}J0"] = curl

    lf = finger_curl(
        lm,
        LITTLE
    )

    fracs["LFJ5"] = float(
        np.clip(
            lf * 0.4,
            0.0,
            1.0
        )
    )

    t_curl = thumb_curl(lm)
    t_spread = thumb_spread(lm)

    if ENABLE_THUMB_ROTATION:
        thumb_rot = thumb_rotation_frac(lm)

        # Inversión simétrica para el pulgar de la mano derecha
        if hand_prefix == "rh":
            fracs["THJ5"] = 1.0 - thumb_rot
        else:
            fracs["THJ5"] = 1.0 - thumb_rot 
    else:
        fracs["THJ5"] = 0.5

    fracs["THJ4"] = float(np.clip(
        0.5 + 0.4 * (0.5 - t_spread),
        0.2,
        0.8
    ))

    fracs["THJ3"] = float(
        np.clip(
            0.5 + (t_spread - 0.5) * 0.2,
            0.0,
            1.0
        )
    )

    fracs["THJ2"] = t_curl
    fracs["THJ1"] = t_curl

    return fracs


def straight_fracs():
    f = {
        "WRJ2": 0.5,
        "WRJ1": 0.5,
        "FRJ": 0.5,
        "THJ5": 0.5,
        "THJ4": 0.5,
        "THJ3": 0.5,
        "THJ2": 0.0,
        "THJ1": 0.0,
        "LFJ5": 0.0
    }

    for p in ("FF", "MF", "RF"):
        f[f"{p}J4"] = 0.5
        f[f"{p}J3"] = 0.0
        f[f"{p}J0"] = 0.0

    f["LFJ4"] = 0.5
    f["LFJ3"] = 0.0
    f["LFJ0"] = 0.0

    return f


STRAIGHT_FRACS = straight_fracs()


class HandActuatorIndex:
    def __init__(self, model, prefix):
        self.idx = {}
        self.lo = {}
        self.hi = {}

        for suffix in STRAIGHT_FRACS:
            name = f"{prefix}_A_{suffix}"

            aid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name
            )

            if aid == -1:
                raise ValueError(
                    f"No existe {name}"
                )

            lo, hi = model.actuator_ctrlrange[aid]

            self.idx[suffix] = aid
            self.lo[suffix] = lo
            self.hi[suffix] = hi

    def apply(self, ctrl, fracs, blend):
        for suffix, frac in fracs.items():
            i = self.idx[suffix]

            target = (
                self.lo[suffix]
                + frac * (
                    self.hi[suffix]
                    - self.lo[suffix]
                )
            )

            ctrl[i] = (
                blend * ctrl[i]
                + (1.0 - blend) * target
            )


def main():
    print(
        f"Cargando modelo: {MODEL_PATH}"
    )

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    hands_idx = {
        "rh": HandActuatorIndex(
            model,
            "rh"
        ),
        "lh": HandActuatorIndex(
            model,
            "lh"
        )
    }

    ctrl = np.zeros(
        model.nu
    )

    for hand in hands_idx.values():
        hand.apply(
            ctrl,
            STRAIGHT_FRACS,
            0.0
        )

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

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara"
        )

    with mujoco.viewer.launch_passive(
        model,
        data
    ) as viewer:

        cam_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "frontal"
        )

        if cam_id != -1:
            viewer.cam.type = (
                mujoco.mjtCamera.mjCAMERA_FIXED
            )

            viewer.cam.fixedcamid = cam_id

        print(
            "Cámara iniciada. Pulsa Q para salir."
        )

        prev_time = time.time()

        while (
            viewer.is_running()
            and cap.isOpened()
        ):

            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(
                frame,
                1
            )

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            result = hands.process(
                rgb
            )

            if result.multi_hand_landmarks:

                for (
                    hand_lm,
                    handedness
                ) in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):

                    lm = [
                        (
                            p.x,
                            p.y,
                            p.z
                        )
                        for p in hand_lm.landmark
                    ]

                    label = (
                        handedness
                        .classification[0]
                        .label
                    )

                    prefix = (
                        "rh"
                        if label == "Right"
                        else "lh"
                    )

                    color = (
                        (0, 255, 0)
                        if prefix == "rh"
                        else (255, 128, 0)
                    )

                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(
                            color=color,
                            thickness=2,
                            circle_radius=2
                        ),
                        mp_draw.DrawingSpec(
                            color=color,
                            thickness=2
                        )
                    )

                    name = (
                        "Derecha -> mano derecha robot"
                        if prefix == "rh"
                        else "Izquierda -> mano izquierda robot"
                    )

                    y = (
                        30
                        if prefix == "rh"
                        else 60
                    )

                    cv2.putText(
                        frame,
                        f"{name} | {palm_side(lm, prefix)}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                    fracs = fracs_from_landmarks(
                        lm,
                        prefix
                    )

                    if ENABLE_FOREARM_ROLL:
                        fracs["FRJ"] = (
                            forearm_roll_frac(
                                lm,
                                prefix
                            )
                        )

                    hands_idx[prefix].apply(
                        ctrl,
                        fracs,
                        SMOOTHING
                    )

            data.ctrl[:] = ctrl

            now = time.time()

            elapsed = (
                now - prev_time
            )

            prev_time = now

            steps = min(
                50,
                max(
                    1,
                    int(
                        round(
                            elapsed
                            / model.opt.timestep
                        )
                    )
                )
            )

            for _ in range(steps):
                mujoco.mj_step(
                    model,
                    data
                )

            viewer.sync()

            cv2.imshow(
                "Camara",
                frame
            )

            if (
                cv2.waitKey(1) & 0xFF
            ) == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()