# NDI → MediaPipe Pose → OSC scripts

Two scripts that read a video feed over **NDI** (instead of a webcam) and stream
pose landmarks out as OSC, same as `mediapipe_body.py` does for a webcam.

- **`ndi_to_osc.py`** — tracks **one** person. Simple and fast.
- **`ndi_to_osc_multi.py`** — tracks **up to several people at once**. Slightly
  more setup (downloads a model file on first run) and a bit heavier on CPU.

Both send one OSC message per frame to `/wek/inputs` as a flat list of floats
(x, y for each landmark, one person after another) — same address Wekinator
expects from the other scripts in this repo.

## Requirements

Install the repo's usual dependencies first (see the main [README](../README.md)),
then make sure `cyndilib` is installed too — it's in
[install/requirements.txt](../install/requirements.txt):

```bash
pip install -r install/requirements.txt
```

You'll also need an NDI source already broadcasting on your network (e.g. the
Raspberry Pi camera stream, or NDI Tools/OBS with an NDI output enabled).

## Usage

Both scripts need at minimum the name of the NDI stream to connect to. You
don't need the exact full name — any distinctive substring works (matching is
case-insensitive):

```bash
python scripts/ndi_to_osc.py --ndi-name "PI-CAM"
```

```bash
python scripts/ndi_to_osc_multi.py --ndi-name "PI-CAM" --num-poses 2
```

If no source matches within `--find-timeout` seconds, the script prints the
list of NDI sources it *did* find, so you can check the name.

Add `--preview` on either script to pop up a window with the tracked skeleton
drawn on top, useful for lining up the camera or debugging tracking quality:

```bash
python scripts/ndi_to_osc.py --ndi-name "PI-CAM" --preview
```

Press `q` or close the window to stop.

## Common options

Both scripts share these:

| Flag | Default | What it does |
|---|---|---|
| `--ndi-name` | *(required)* | NDI source name or substring to connect to |
| `--osc-host` | `127.0.0.1` | Where to send OSC messages |
| `--osc-port` | `9000` | OSC port (matches Wekinator's input port in this repo) |
| `--width` / `--height` | `640` / `342` | Resolution frames are downscaled to before tracking (`0 0` disables) |
| `--max-fps` | `30` | Cap processing rate to reduce CPU load (`0` = uncapped, run as fast as possible) |
| `--find-timeout` | `10.0` | Seconds to wait while looking for the NDI source |
| `--preview` | off | Show an annotated preview window |
| `--color-format` | `fastest` | NDI pixel format to request — see below |
| `--bandwidth` | `lowest` | NDI bandwidth mode — bump to `highest` if frames look corrupted |

`ndi_to_osc_multi.py` also has:

| Flag | Default | What it does |
|---|---|---|
| `--num-poses` | `2` | Max number of people to track at once |
| `--model-complexity` | `1` | `0` lite / `1` full / `2` heavy — bigger = more accurate, slower |
| `--gpu` | off | Run pose inference on the GPU instead of the CPU (needs a working OpenGL/EGL driver) |

`ndi_to_osc.py` also has `--model-complexity` (same `0`/`1`/`2` meaning), but
uses it to configure MediaPipe's single-person model directly rather than
picking a downloadable model file.

## Notes

- `ndi_to_osc_multi.py` downloads its pose model (a few MB) into
  `scripts/models/` the first time it runs with a given `--model-complexity`.
  That folder is git-ignored — it'll just re-download if it's missing.
- If tracking feels laggy, try lowering `--width`/`--height` before reaching
  for a lower `--model-complexity` — resolution tends to matter more.
- `ndi_to_osc_multi.py` re-detects all poses from scratch on every frame
  (MediaPipe's IMAGE mode) instead of tracking between frames, specifically
  so it reliably picks up more than one person — MediaPipe's streaming modes
  tend to lock onto a single tracked person and largely ignore `--num-poses`.
  This trades away a bit of the smoothing/jitter-reduction you'd get from
  frame-to-frame tracking, but that's the tradeoff to actually get multiple
  bodies recognized.
- `--gpu` on `ndi_to_osc_multi.py` roughly halved inference time in local
  testing (integrated Intel GPU via Mesa: ~38ms/frame CPU vs ~20ms/frame
  GPU). Actual gains depend on your GPU/drivers. If usage still looks
  CPU-only with `--gpu` set, MediaPipe can silently fall back to CPU without
  raising an error — check the terminal output when the script starts for a
  line like `Created TensorFlow Lite delegate for GPU`; if you don't see it
  (or see a fallback warning instead), your driver setup doesn't support the
  GPU delegate here. `ndi_to_osc.py` (the single-person script) has no GPU
  option at all — MediaPipe's legacy `solutions.pose` API it's built on
  doesn't expose one in Python; only the newer Tasks API does.
- **CPU usage**: the NDI receive thread polls for frames in a loop that, by
  itself, isn't rate-limited — `capture_video()` just hands back whatever's
  currently buffered, even if unchanged, so an unthrottled loop can spin far
  faster than the camera's actual frame rate for no benefit. `--max-fps`
  (default `30`, matching the typical NDI source frame rate) caps both the
  NDI polling loop and the pose-inference loop so they don't do that. This
  costs no real freshness — you can't get frames faster than the camera
  produces them — but cuts a lot of wasted CPU. If you still need more
  headroom for other software, lower `--max-fps` further (e.g. `15`), drop
  `--width`/`--height`, or use `--model-complexity 0`.
