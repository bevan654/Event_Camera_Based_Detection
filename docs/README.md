# Docs

Example images made while exploring the ev-CIVIL field dataset (DAVIS346 event camera, 346×260).

| File | What it shows |
|---|---|
| `pipeline_flowchart.svg` | The overall pipeline: events → time window → ON/OFF grids → image → detector → crack boxes. |
| `dataset_preview_frames_vs_events.png` | Four recordings side by side: raw camera frame, the same frame with CLAHE contrast boost, and the events from a fixed 25 ms (day) / 40 ms (challenge) window. Red = ON (got brighter), green = OFF (got darker). Boxes: red = crack, pink = spalling. Rows: `dset_1/crack_23` (clean), `dset_1/crack_40` (several cracks, some unlabelled), `day_time_challenge/spalling_31` (bright, busy scene), `day_time_challenge/crack_89` (dark, noisy). |
| `event_image_example.png` | One event image built with the paper's adaptive window (Algorithm 1) from `dset_1/crack_23`: 1000 events over ~22 ms. Shows the camera frame for reference, each channel on its own (ON, OFF, ON+OFF), the combined 3-channel image that YOLO sees, and the crack box. |
| `event_image_grayscale_options.png` | The same event image as 3-channel (displayed as colour) vs two grayscale alternatives: A = ON+OFF, B = gray background with ON white / OFF black. |

Note: the colours in event images are only how a viewer displays 3 numbers per pixel — the model sees the numbers, not colours.
