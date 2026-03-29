MaixCam2 Camera Passthrough (JPEG Stream)
====

This example is for camera passthrough only:
- MaixCam2 captures camera frames.
- MaixCam2 streams JPEG frames over HTTP.
- PC pulls the stream and performs all CV compute and actuator control.

Run
----

- `./stream_jpg_demo --width 640 --height 480 --fps 30 --port 8000 --no-display`
- Legacy positional mode is still supported: `./stream_jpg_demo 640 480 0 30 3`

Open stream URL
----

- `http://<maixcam_ip>:8000/stream`

Arguments
----

- `--width N` camera width
- `--height N` camera height
- `--fps N` camera fps
- `--buffer N` camera buffer number
- `--port N` HTTP stream port
- `--display` show local display on device
- `--no-display` disable local display (default)

PC-side processing idea
----

Use OpenCV on PC to read the stream URL and run detection/tracking locally,
then send control commands directly from PC to your actuator interface.

