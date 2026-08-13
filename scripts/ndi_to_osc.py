import argparse
import threading
import time

import cv2
import mediapipe as mp

from cyndilib.finder import Finder
from cyndilib.receiver import Receiver
from cyndilib.video_frame import VideoFrameSync
from cyndilib.wrapper import RecvBandwidth, RecvColorFormat
from cyndilib.wrapper.ndi_structs import FourCC

from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

mp_pose = mp.solutions.pose
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


def send_pose_as_single_osc(landmarks):
    coordinates = []
    for landmark in landmarks:
        coordinates.extend([1.0 - float(landmark.x), 1.0 - float(landmark.y)])

    osc_msg = oscbuildparse.OSCMessage("/wek/inputs", None, coordinates)
    osc_send(osc_msg, "localhost")


def pose_loop(buf, stop_event, model_complexity, preview):
    with mp_pose.Pose(
        smooth_landmarks=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
    ) as pose:
        while not stop_event.is_set():
            rgb = buf.get()
            if rgb is None:
                time.sleep(0.001)
                continue

            rgb.flags.writeable = False
            results = pose.process(rgb)

            if results.pose_landmarks:
                osc_process()
                send_pose_as_single_osc(results.pose_landmarks.landmark)

            if preview:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                bgr = cv2.flip(bgr, 1)
                cv2.imshow("frame", bgr)
                if cv2.waitKey(1) == ord("q") or cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break

    if preview:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NDI -> MediaPipe Pose -> OSC")
    parser.add_argument("--ndi-name", required=True, help="NDI source name (or substring) to connect to")
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument("--width", type=int, default=640, help="Downscale width before pose inference (0 to disable)")
    parser.add_argument("--height", type=int, default=342, help="Downscale height before pose inference (0 to disable)")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--color-format", default="fastest", choices=sorted(COLOR_FORMATS))
    parser.add_argument("--bandwidth", default="lowest", choices=sorted(BANDWIDTHS))
    parser.add_argument("--find-timeout", type=float, default=10.0, help="Seconds to wait while discovering the NDI source")
    parser.add_argument("--preview", action="store_true", help="Show an annotated preview window (adds overhead)")
    args = parser.parse_args()

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
        pose_loop(buf, stop_event, args.model_complexity, args.preview)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ndi_thread.join(timeout=2.0)
        receiver.disconnect()
        osc_terminate()


if __name__ == "__main__":
    main()
