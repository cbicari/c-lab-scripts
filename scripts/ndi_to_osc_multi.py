import argparse
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync
from cyndilib.wrapper import RecvBandwidth, RecvColorFormat
from cyndilib.wrapper.ndi_structs import FourCC

from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

mp_pose_sol = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

COLOR_FORMATS = {
    "fastest": RecvColorFormat.fastest,
    "best": RecvColorFormat.best,
    "uyvy_bgra": RecvColorFormat.UYVY_BGRA,
    "uyvy_rgba": RecvColorFormat.UYVY_RGBA,
    "bgrx_bgra": RecvColorFormat.BGRX_BGRA,
    "rgbx_rgba": RecvColorFormat.RGBX_RGBA,
}
BANDWIDTHS = {
    "lowest": RecvBandwidth.lowest,
    "highest": RecvBandwidth.highest,
}
# NDI SDK's `bits_per_pixel` reports 24 for the X (no-alpha) variants even
# though the buffer is still packed 4 bytes/pixel; use nominal sizes instead.
BYTES_PER_PIXEL = {
    FourCC.RGBA: 4,
    FourCC.BGRA: 4,
    FourCC.RGBX: 4,
    FourCC.BGRX: 4,
    FourCC.UYVY: 2,
}

MODEL_VARIANTS = {0: "lite", 1: "full", 2: "heavy"}
MODEL_DIR = Path(__file__).parent / "models"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)


def ensure_model(variant):
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        url = MODEL_URL.format(variant=variant)
        print(f"Downloading {variant} pose landmarker model from {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def find_source(finder, name, timeout):
    deadline = time.monotonic() + timeout
    while True:
        finder.update_sources()
        for source in finder.iter_sources():
            if name.lower() in source.name.lower():
                return source
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"No NDI source matching {name!r} found after {timeout:.0f}s. "
                f"Available sources: {finder.get_source_names()}"
            )
        finder.wait_for_sources(min(0.5, remaining))


def frame_to_rgb(video_frame):
    xres, yres = video_frame.get_resolution()
    if xres <= 0 or yres <= 0:
        return None

    fourcc = video_frame.fourcc
    bpp = BYTES_PER_PIXEL.get(fourcc)
    if bpp is None:
        raise ValueError(f"Unsupported NDI fourcc for pose input: {fourcc!r}")

    stride = video_frame.get_line_stride()
    packed = video_frame.get_array().reshape(yres, stride)[:, : xres * bpp]
    frame = packed.reshape(yres, xres, bpp)

    if fourcc in (FourCC.RGBA, FourCC.RGBX):
        return frame[:, :, :3]
    if fourcc in (FourCC.BGRA, FourCC.BGRX):
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_UYVY)  # UYVY


class LatestFrame:
    """Single-slot, overwrite-on-write buffer shared between the NDI and pose threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def put(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def ndi_receive_loop(receiver, buf, stop_event, width, height):
    frame_sync = receiver.frame_sync
    while not stop_event.is_set():
        frame_sync.capture_video()
        rgb = frame_to_rgb(frame_sync.video_frame)
        if rgb is None:
            time.sleep(0.001)
            continue
        if width and height:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        buf.put(rgb)


def send_poses_as_single_osc(poses_landmarks):
    coordinates = []
    for landmarks in poses_landmarks:
        for landmark in landmarks:
            coordinates.extend([1.0 - float(landmark.x), 1.0 - float(landmark.y)])

    osc_msg = oscbuildparse.OSCMessage("/wek/inputs", None, coordinates)
    osc_send(osc_msg, "localhost")


def pose_loop(buf, stop_event, model_path, num_poses, preview):
    # IMAGE mode, not VIDEO/LIVE_STREAM: MediaPipe's streaming modes track a
    # single pose across frames and only reach for the multi-person detector
    # when tracking is lost, so num_poses > 1 is unreliable there. IMAGE mode
    # re-runs full multi-person detection on every frame - no cross-frame
    # tracking state, but that's what actually finds more than one person.
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while not stop_event.is_set():
            rgb = buf.get()
            if rgb is None:
                time.sleep(0.001)
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                osc_process()
                send_poses_as_single_osc(result.pose_landmarks)

            if preview:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for landmarks in result.pose_landmarks:
                    proto = landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend(
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                        for lm in landmarks
                    )
                    mp_draw.draw_landmarks(bgr, proto, mp_pose_sol.POSE_CONNECTIONS)
                bgr = cv2.flip(bgr, 1)
                cv2.imshow("frame", bgr)
                if cv2.waitKey(1) == ord("q") or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break

    if preview:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NDI -> MediaPipe Pose (multi-person) -> OSC")
    parser.add_argument("--ndi-name", required=True, help="NDI source name (or substring) to connect to")
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--width", type=int, default=640, help="Downscale width before pose inference (0 to disable)")
    parser.add_argument("--height", type=int, default=342, help="Downscale height before pose inference (0 to disable)")
    parser.add_argument("--num-poses", type=int, default=2, help="Maximum number of people to track at once")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="0=lite, 1=full, 2=heavy")
    parser.add_argument("--color-format", default="fastest", choices=sorted(COLOR_FORMATS))
    parser.add_argument("--bandwidth", default="lowest", choices=sorted(BANDWIDTHS))
    parser.add_argument("--find-timeout", type=float, default=10.0, help="Seconds to wait while discovering the NDI source")
    parser.add_argument("--preview", action="store_true", help="Show an annotated preview window (adds overhead)")
    args = parser.parse_args()

    model_path = ensure_model(MODEL_VARIANTS[args.model_complexity])

    finder = Finder()
    finder.open()
    try:
        print(f"Looking for NDI source matching {args.ndi_name!r}...")
        source = find_source(finder, args.ndi_name, args.find_timeout)
        print(f"Connecting to NDI source: {source.name}")

        receiver = Receiver(
            color_format=COLOR_FORMATS[args.color_format],
            bandwidth=BANDWIDTHS[args.bandwidth],
        )
        # Receiver(source=...) only records the source, it never connects -
        # connect_to() must be called explicitly, and the Finder must still
        # be open since it owns the Source's underlying NDI pointer.
        receiver.connect_to(source)
    finally:
        finder.close()

    receiver.frame_sync.set_video_frame(VideoFrameSync())

    buf = LatestFrame()
    stop_event = threading.Event()

    osc_startup()
    osc_udp_client(args.osc_host, args.osc_port, "localhost")

    ndi_thread = threading.Thread(
        target=ndi_receive_loop,
        args=(receiver, buf, stop_event, args.width, args.height),
        daemon=True,
    )
    ndi_thread.start()

    try:
        pose_loop(buf, stop_event, model_path, args.num_poses, args.preview)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ndi_thread.join(timeout=2.0)
        receiver.disconnect()
        osc_terminate()


if __name__ == "__main__":
    main()
