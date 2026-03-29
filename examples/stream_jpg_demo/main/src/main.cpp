
#include "maix_jpg_stream.hpp"
#include "maix_basic.hpp"
#include "maix_vision.hpp"
#include "main.h"

#include <memory>
#include <string>

using namespace maix;
extern std::string html;

int _main(int argc, char* argv[])
{
    int cam_w = 640;
    int cam_h = 480;
    image::Format cam_fmt = image::Format::FMT_RGB888;
    int cam_fps = 30;
    int cam_buffer_num = 3;
    int http_port = 8000;
    bool enable_display = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            log::info("Usage:\n"
                      "  ./stream_jpg_demo [--width N] [--height N] [--fps N] [--port N] [--display|--no-display]\\n"
                      "  ./stream_jpg_demo <width> <height> <format> <fps> <buff_num>  (legacy positional)\\n"
                      "Examples:\n"
                      "  ./stream_jpg_demo --width 640 --height 480 --fps 30 --port 8000 --no-display\\n"
                      "  ./stream_jpg_demo 640 480 0 30 3");
            return 0;
        } else if (arg == "--width" && i + 1 < argc) {
            cam_w = atoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            cam_h = atoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            cam_fps = atoi(argv[++i]);
        } else if (arg == "--port" && i + 1 < argc) {
            http_port = atoi(argv[++i]);
        } else if (arg == "--buffer" && i + 1 < argc) {
            cam_buffer_num = atoi(argv[++i]);
        } else if (arg == "--display") {
            enable_display = true;
        } else if (arg == "--no-display") {
            enable_display = false;
        } else if (!arg.empty() && arg[0] != '-') {
            // Backward compatibility: positional args.
            cam_w = atoi(argv[i]);
            if (i + 1 < argc) cam_h = atoi(argv[i + 1]);
            if (i + 2 < argc) cam_fmt = (image::Format)atoi(argv[i + 2]);
            if (i + 3 < argc) cam_fps = atoi(argv[i + 3]);
            if (i + 4 < argc) cam_buffer_num = atoi(argv[i + 4]);
            break;
        }
    }

    cam_w = std::max(1, cam_w);
    cam_h = std::max(1, cam_h);
    cam_fps = std::max(1, cam_fps);
    cam_buffer_num = std::max(1, cam_buffer_num);
    http_port = std::max(1, http_port);

    log::info("Camera passthrough mode");
    log::info("camera width:%d height:%d format:%s fps:%d buffer_num:%d", cam_w, cam_h, image::fmt_names[cam_fmt].c_str(), cam_fps, cam_buffer_num);
    log::info("stream port:%d local_display:%d", http_port, enable_display ? 1 : 0);

    camera::Camera cam = camera::Camera(cam_w, cam_h, cam_fmt, "", cam_fps, cam_buffer_num);
    std::unique_ptr<display::Display> disp;
    if (enable_display) {
        disp = std::make_unique<display::Display>();
    }

    log::info("camera open success");
    log::info("camera size: %dx%d\n", cam.width(), cam.height());
    if (disp) {
        log::info("disp size: %dx%d\n", disp->width(), disp->height());
    }

    http::JpegStreamer stream = http::JpegStreamer("", http_port);
    stream.set_html(html);
    stream.start();

    log::info("stream ready: http://%s:%d/stream\r\n", stream.host().c_str(), stream.port());
    uint64_t frame_count = 0;
    uint64_t t_last = time::ticks_ms();
    while(!app::need_exit())
    {
        // read image from camera
        image::Image *img = cam.read();
        err::check_null_raise(img, "camera read failed");

        image::Image *jpg = img->to_jpeg();
        stream.write(jpg);
        delete jpg;

        if (disp) {
            disp->show(*img);
        }

        frame_count++;
        uint64_t t_now = time::ticks_ms();
        if (t_now - t_last >= 2000) {
            const float fps = frame_count * 1000.0f / static_cast<float>(t_now - t_last);
            log::info("stream fps: %.2f", fps);
            frame_count = 0;
            t_last = t_now;
        }

        // free image data, important!
        delete img;
    }
    stream.stop();
    return 0;
}

std::string html =
"<!DOCTYPE html>\n"
"<html>\n"
"<head>\n"
"    <title>MaixCam2 Camera Passthrough</title>\n"
"</head>\n"
"<body>\n"
"    <h1>MaixCam2 Camera Passthrough</h1>\n"
"    <p>This device only streams camera frames. Do all CV and control on PC side.</p>\n"
"    <img src=\"/stream\" alt=\"Stream\">\n"
"</body>\n"
"</html>\n";


int main(int argc, char* argv[])
{
    // Catch signal and process
    sys::register_default_signal_handle();

    // Use CATCH_EXCEPTION_RUN_RETURN to catch exception,
    // if we don't catch exception, when program throw exception, the objects will not be destructed.
    // So we catch exception here to let resources be released(call objects' destructor) before exit.
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}


