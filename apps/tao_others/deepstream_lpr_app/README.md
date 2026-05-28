# Sample For Car License Recognization
 - [Description](#description)
 - [Performance](#performance)
 - [Prerequisition](#prerequisition)
 - [Download](#download)
 - [Prepare Triton Server](#prepare-triton-server)
 - [Build and Run](#build-and-run)
 - [Notice](#notice)

---

## Description
This sample is to show how to use graded models for detection and classification with DeepStream SDK version not less than 5.0.1. The models in this sample are all TAO3.0 models.

`PGIE(car detection) -> SGIE(car license plate detection) -> SGIE(car license plate recognization)`

The app now also supports optional vehicle attribute classifiers for color, type, and make. When the launch YAML includes `secondary-gie2`, `secondary-gie3`, and/or `secondary-gie4`, their labels are attached to the resolved vehicle object and flow into:

- `events.jsonl` and debug JSONL entries
- live dashboard events served by `app.py`
- generated `index.csv` and `index.html` case review output

![LPR/LPD application](lpr.png)

This pipeline is based on three TAO models below

* Vehicle detection model https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet
* LPD (car license plate detection) model https://ngc.nvidia.com/catalog/models/nvidia:tao:lpdnet
* LPR (car license plate recognization/text extraction) model https://ngc.nvidia.com/catalog/models/nvidia:tao:lprnet

More details for TAO3.0 LPD and LPR models and TAO training, please refer to [TAO document](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html).

## Performance
Below table shows the end-to-end performance of processing 1080p videos with this sample application.

| Device    | Number of streams | Batch Size | Total FPS |
|-----------| ----------------- | -----------|-----------|
|Jetson Nano|     1             |     1      | 9.2       |
|Jetson NX  |     3             |     3      | 80.31     |
|Jetson Xavier |  5             |     5      | 146.43    |
|Jetson Orin|     5             |     5      | 341.65    |
|T4         |     14            |     14     | 447.15    |

## Prerequisition

* [DeepStream SDK 6.0 or above](https://developer.nvidia.com/deepstream-getting-started)

  Make sure deepstream-test1 sample can run successful to verify your DeepStream installation

* [tao-converter](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter/version)

  Download x86 or Jetson tao-converter which is compatible to your platform from the links in https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter/version.
* [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
 
  The LPR sample application can work as Triton client on x86 platforms.

## Download

1. Download Project with SSH or HTTPS

```shell
    # SSH
    git clone git@github.com:NVIDIA-AI-IOT/deepstream_tao_apps.git
    # or HTTPS
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps.git
```

2. Prepare Models

All models can be downloaded with the following commands:

```shell
    cd deepstream_tao_apps/
    ./download_models.sh
```

## Prepare Triton Server
From DeepStream 6.1, LPR sample application supports three inferencing modes:
* gst-nvinfer inferencing based on TensorRT
* gst-nvinferserver inferencing as Triton CAPI client(only for x86)
* gst-nvinferserver inferencing as Triton gRPC client(only for x86)

The following instructions are only needed for the LPR sample application working with gst-nvinferserver inferencing on x86 platforms as the Triton client. For LPR sample application works with nvinfer mode, please go to [Build and Run](#build-and-run) part directly.

The Triton Inference Server libraries are required to be installed if the DeepStream LPR sample application should work as the Triton client, the Triton client [document](https://github.com/triton-inference-server/client) instructs how to install the necessary libraries. A easier way is to run DeepStream application in the [DeepStream Triton container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream).

* Setting up Triton Inference Server for native cAPI inferencing, please refer to [triton_server.md](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server.md).

* Setting up Triton Inference Server for gRPC inferencing, please refer to [triton_server_grpc.md](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/blob/master/triton_server_grpc.md). 

## Build and Run

```shell
    # Build
    cd apps/tao_others/deepstream_lpr_app
    make
```

A sample of US car plate recognition:

```shell
    cp dict_us.txt dict.txt
    # nvinfer is used by default, please modify the configuration file 
    # for nvinferserver capi/nvinferserver grpc
    ./deepstream-lpr-app ../../../configs/app/lpr_app_us_config.yml
```

A sample of Chinese car plate recognition:

```shell
    cp dict_ch.txt dict.txt
    # nvinfer is used by default, please modify the configuration file 
    # for nvinferserver capi/nvinferserver grpc
    ./deepstream-lpr-app ../../../configs/app/lpr_app_ch_config.yml
```

Case review index generation:

```shell
    python3 build_case_index.py \
      /home/jason/Projects/alpr-deepstream/deepstream_tao_apps/apps/tao_others/deepstream_lpr_app/evidence/case_2026_04_07_016

    xdg-open \
      /home/jason/Projects/alpr-deepstream/deepstream_tao_apps/apps/tao_others/deepstream_lpr_app/evidence/case_2026_04_07_016/index.html
```

The application also generates `index.html` and `index.csv` automatically at the end of each run for the active case directory.

Live dashboard service:

```shell
    python3 -m pip install fastapi uvicorn pydantic

    uvicorn app:app --host 0.0.0.0 --port 8080
```

Then browse from the field unit itself or another device on the same hotspot/network:

```shell
    http://127.0.0.1:8080
    http://<field-unit-ip>:8080
```

  When the FastAPI service is reachable, the C++ ALPR pipeline now publishes real `CONFIRMED` and `LOCKED` events to `/api/live-event` automatically as evidence is written.

Remote UI access over WiFi AP or Ethernet:

The backend already listens on all interfaces by default:

```shell
  ALPR_DASHBOARD_HOST=0.0.0.0
  ALPR_DASHBOARD_PORT=8080
```

To see which URLs are currently reachable on the field unit:

```shell
  ./scripts/configure_network_access.sh status
```

Bring up a local WiFi access point with NetworkManager:

```shell
  sudo ./scripts/configure_network_access.sh hotspot-up wlp3s0 ALPR-Field-Unit ALPRAccess123
```

Then connect a phone, tablet, or laptop to that SSID and open the URL printed by the script. In most shared-mode NetworkManager setups this will be the hotspot IP on that WiFi adapter, typically something like:

```text
  http://10.42.0.1:8080
```

Bring up a direct Ethernet shared link with DHCP for a laptop:

```shell
  sudo ./scripts/configure_network_access.sh ethernet-up enp2s0
```

Then connect the laptop to the field unit with an Ethernet cable and open the URL printed by the script.

Tear either network helper down when finished:

```shell
  sudo ./scripts/configure_network_access.sh hotspot-down
  sudo ./scripts/configure_network_access.sh ethernet-down
```

Notes:

- The helper script requires `nmcli` and NetworkManager.
- A WiFi adapter usually cannot stay connected to an upstream WiFi network and run a hotspot at the same time unless the hardware/driver supports concurrent AP + station mode.
- The live UI and config UI now show the currently detected access URLs in their status panels.

Optional vehicle attribute classifiers:

```yaml
secondary-gie2:  # vehicle color classifier
secondary-gie3:  # vehicle type classifier
secondary-gie4:  # vehicle make classifier
```

These sections are optional. If omitted, the app still runs the existing ALPR pipeline and the dashboard/index simply leave `vehicle_color`, `vehicle_type`, and `vehicle_make` blank. If present, ensure the classifier configs use unique IDs `4`, `5`, and `6` respectively so the app can read the metadata correctly.

  Default live publisher behavior:

  ```shell
    ALPR_LIVE_ENDPOINT=http://127.0.0.1:8080/api/live-event
  ```

  Config-based live dashboard settings now live in the app YAML under `live-dashboard:`. For example in `configs/app/lpr_app_us_config.yml`:

  ```yaml
  source-list:
    list: v4l2:///dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0

  live-dashboard:
    endpoint: http://127.0.0.1:8080/api/live-event
    source-map:
      source_0: RF
  ```

  For the current USB-camera test setup, the ArduCam is exposed on `/dev/video0`, but the safer source entry is the stable `/dev/v4l/by-path/...` symlink for the USB port shown above so the app does not drift to the integrated webcam when `/dev/video*` numbering changes or the camera's by-id name changes after reconnect.

  ```text
    source_0 -> RF
  ```

  The startup script now reads `live-dashboard.source-map` from the selected YAML and exports it for both the backend and the DeepStream process, so the dashboard status row and live events stay aligned.

Quick live-event test:

```shell
    curl -X POST http://127.0.0.1:8080/api/live-event \
      -H "Content-Type: application/json" \
      -d '{
        "event_id": "case_2026_04_07_022_evt_000021",
        "case_id": "case_2026_04_07_022",
        "plate": "MWP2083",
        "status": "LOCKED",
        "confidence": 85,
        "source": "LF",
        "source_label": "Left Front",
        "timestamp_utc": "2026-04-07T17:19:41Z",
        "frame_number": 458,
        "track_id": 44,
        "track_id_valid": true,
        "full_frame_path": "case_2026_04_07_022/frames/case_2026_04_07_022_evt_000021_locked_full.jpg",
        "annotated_frame_path": "case_2026_04_07_022/frames/case_2026_04_07_022_evt_000021_locked_annotated.jpg",
        "plate_crop_path": "case_2026_04_07_022/frames/case_2026_04_07_022_evt_000021_locked_plate.jpg"
      }'

    curl http://127.0.0.1:8080/api/live-events
    curl http://127.0.0.1:8080/api/status
```

For `/media/{relative_path}`, store event image paths relative to the evidence root. For example, prefer `case_2026_04_07_022/frames/...` over bare `frames/...` so the dashboard can browse evidence across cases.

Boot/run discipline:

```shell
  ./scripts/start_field_unit.sh
  ./scripts/stop_field_unit.sh
```

Optional config override:

```shell
  ./scripts/start_field_unit.sh /absolute/path/to/lpr_app_us_config.yml
```

Optional boot-time network access:

If you run the systemd service as root, you can enable hotspot and/or Ethernet sharing automatically with `/etc/default/alpr-field-unit`:

Use [systemd/alpr-field-unit.env.example](systemd/alpr-field-unit.env.example) as the starting point for that file.

```shell
  ALPR_DASHBOARD_HOST=0.0.0.0
  ALPR_DASHBOARD_PORT=8080

  ALPR_WIFI_AP_ENABLE=1
  ALPR_WIFI_AP_INTERFACE=wlp3s0
  ALPR_WIFI_AP_SSID=ALPR-Field-Unit
  ALPR_WIFI_AP_PASSWORD=ALPRAccess123

  ALPR_ETHERNET_SHARE_ENABLE=1
  ALPR_ETHERNET_INTERFACE=enp2s0
```

Only enable the interfaces you actually want to manage from the service.

Known runtime locations:

```text
  logs/backend.log
  logs/alpr.log
  runtime/backend.pid
  runtime/alpr.pid
  runtime/alpr_status.json
```

The dashboard status panel reads `runtime/alpr_status.json` plus the pid files to show ALPR alive, current case, source activity, storage free, and last event time. That same layout is meant to be the handoff point for a later systemd unit.

Systemd handoff:

```shell
  cd systemd
  sudo cp alpr-field-unit.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable alpr-field-unit.service
  sudo systemctl start alpr-field-unit.service
```

The provided unit uses the existing start/stop scripts so the runtime, pid, and log locations remain unchanged.

Hotlist integration:

Place source files in `hotlists/` using these names:

```text
  hotlists/svs.tbl
  hotlists/slr.tbl
  hotlists/sfr.tbl
```

Each file should look like:

```text
  DATE04/07/2026 11:00
  9ASX920 CA0120230108
  4ZOG280 CA1920241122
  8NQT723 CA1920260406
```

The backend normalizes records into plate, state, county code, entry date, list type, list label, and source file. v1 matching is exact uppercase plate text only, so multiple-state collisions remain possible; when that happens, the event keeps all matching records and marks the highest-priority hit as `SFR > SVS > SLR`.

Hotlist management UI:

- Open `/hotlists` from the live dashboard or camera config page.
- Upload updated `svs.tbl`, `slr.tbl`, and `sfr.tbl` files directly from a phone, tablet, or laptop on the same hotspot/network.
- Add single local entries manually without changing the agency source files.
- Import extra local entries from CSV. Supported CSV columns are `plate`, `list_type`, `state`, `county_code`, and `entry_date`. If no header is present, the first column is treated as the plate and the selected default list type is used.
- The effective hotlist is the merge of the agency files plus the local overlay.
- Runtime state is persisted in `hotlists/hotlist_manifest.json` and `hotlists/local_hotlist_entries.json` and is intentionally ignored by git.

Operator alert policy:

- Live dashboard sightings are grouped by tracked target, so `CONFIRMED` and `LOCKED` updates collapse into one evolving card instead of separate duplicate rows.
- The hotlist check runs against the best read currently known for that grouped sighting, preferring `LOCKED` over `CONFIRMED` and higher confidence over lower confidence.
- Only grouped sightings whose current best read is `LOCKED` can trigger the top-of-screen alert strip and hotlist tone.
- The operator can acknowledge a live hotlist alert, which suppresses that plate for 30 minutes by default.
- As a safety fallback, if the same hotlist plate produces 5 `LOCKED` sightings inside 60 seconds, the backend auto-snoozes that plate for the same 30-minute window.
- Set `HOTLIST_ACKNOWLEDGE_SECONDS`, `HOTLIST_AUTO_SNOOZE_COUNT`, and `HOTLIST_AUTO_SNOOZE_WINDOW_SECONDS` in the backend environment to adjust the suppression behavior.

Hotlist endpoints:

```shell
  curl http://127.0.0.1:8080/api/hotlist/status
  curl http://127.0.0.1:8080/api/hotlist/entries?source_kind=local
  curl -X POST http://127.0.0.1:8080/api/hotlist/reload
  curl -X POST http://127.0.0.1:8080/api/hotlist/local-entry \
    -H 'Content-Type: application/json' \
    -d '{"plate":"ABC1234","list_type":"SLR","state":"CA","county_code":"00"}'
```

When a live sighting matches, the dashboard shows a priority-colored hotlist badge (`SFR`, `SVS`, or `SLR`) plus the highest-priority label and a match count if multiple records exist for the same plate. The dashboard keeps one grouped card per sighting and updates that card as the read improves, so operators see the current best read rather than a separate `CONFIRMED` row followed by a separate `LOCKED` row. Backend-approved hotlist alerts are pinned into a dedicated alert strip at the top of the page for 60 seconds, where operators can acknowledge them to quiet that plate without removing the underlying event card from the feed. Grouped cards that are still waiting for `LOCKED`, or are inside the quiet period after acknowledgement or auto-snooze, stay visible with status text but do not retrigger the strip or the hotlist tone. The live UI also includes a `Hotlist Hits` filter, uses a distinct audio tone for approved hotlist alerts so they stand out from normal `LOCKED` events, and provides a `Test Alert` button so the operator can explicitly unlock and verify audio on the target device.

Acknowledgement endpoint:

```shell
  curl -X POST http://127.0.0.1:8080/api/hotlist/acknowledge \
    -H 'Content-Type: application/json' \
    -d '{"plate":"MWP2083","seconds":1800}'
```

## Notice
1. This sample application only support mp4 files which contain H264 videos as input files.
2. For Chinese plate recognition, please make sure the OS supports Chinese language.
3. The second argument of deepstream-lpr-app should be 2(fakesink) for performance test.
4. The DashCamNet primary detector and LPD model run as TAO/DeepStream detector stages, and the LPR model is FP16.
5. There is a bug for Triton gprc mode: the first two character can't be recognized.
6. For some yolo models, some layers of the models should use FP32 precision. This is a network characteristics that the accuracy drops rapidly when maximum layers are run in INT8 precision. Please refer the [layer-device-precision](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) for more details.
7. For Chinese plate recognition, please make sure the Chinese language support is in the OS. `Take Ubuntu as an example :`

 - Install Chinese Language package . 
   ```bash
   sudo apt-get install language-pack-zh-hans
   ```

 - Set the Chinese language enviroment
   ```bash
   export LANG=zh_CN.UTF-8
   export LANGUAGE="zh_CN:zh:en_US:en"
   ```
