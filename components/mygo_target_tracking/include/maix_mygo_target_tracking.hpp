#pragma once

#include "TargetTracking/TargetTracker.hpp"
#include "TargetTracking/TargetTrackingPipeline.hpp"
#include "maix_image.hpp"

#ifndef MYGO_TARGETTRACKING_USE_MAIX
#include "maix_image_cv.hpp"
#endif

/**
 * @brief Mygo target tracking API wrapper for MaixCDK/MaixPy.
 * @maixpy maix.mygo_target_tracking
 */
namespace maix::mygo_target_tracking
{
    /**
     * @brief Detection result in angle and pixel domains.
     * @maixpy maix.mygo_target_tracking.DetectResult
     */
    class DetectResult
    {
    public:
        /**
         * @brief Whether target is found.
         * @maixpy maix.mygo_target_tracking.DetectResult.found
         */
        bool found = false;

        /**
         * @brief Board center x in pixels.
         * @maixpy maix.mygo_target_tracking.DetectResult.board_cx
         */
        float board_cx = -1.0f;

        /**
         * @brief Board center y in pixels.
         * @maixpy maix.mygo_target_tracking.DetectResult.board_cy
         */
        float board_cy = -1.0f;

        /**
         * @brief Target center x in pixels.
         * @maixpy maix.mygo_target_tracking.DetectResult.target_cx
         */
        float target_cx = -1.0f;

        /**
         * @brief Target center y in pixels.
         * @maixpy maix.mygo_target_tracking.DetectResult.target_cy
         */
        float target_cy = -1.0f;

        /**
         * @brief Distance from board center to target center in pixels.
         * @maixpy maix.mygo_target_tracking.DetectResult.distance
         */
        float distance = 0.0f;

        /**
         * @brief Angle from board center to target center in degrees.
         * @maixpy maix.mygo_target_tracking.DetectResult.angle
         */
        float angle = 0.0f;

        /**
         * @brief Pitch error mapped from image to angle in degrees.
         * @maixpy maix.mygo_target_tracking.DetectResult.pitch_error
         */
        float pitch_error = 0.0f;

        /**
         * @brief Yaw error mapped from image to angle in degrees.
         * @maixpy maix.mygo_target_tracking.DetectResult.yaw_error
         */
        float yaw_error = 0.0f;
    };

    /**
     * @brief Wrapper class for Mygo target recognition.
     *
     * This class keeps original TargetTracker logic and exposes a stable API for MaixPy.
     * @maixpy maix.mygo_target_tracking.Recognizer
     */
    class Recognizer
    {
    public:
        /**
         * @brief Construct a recognizer.
         * @maixpy maix.mygo_target_tracking.Recognizer.__init__
         */
        Recognizer() {}

        /**
         * @brief Enable/disable debug output.
         * @param print_debug Print debug logs.
         * @param show_debug Show debug windows when platform supports GUI.
         * @maixpy maix.mygo_target_tracking.Recognizer.set_debug
         */
        void set_debug(bool print_debug = false, bool show_debug = false)
        {
            TrackerConfig cfg = tracker_.get_config();
            cfg.print_debug_info = print_debug;
            cfg.show_debug_windows = show_debug;
            tracker_.set_config(cfg);
        }

        /**
         * @brief Configure key blob filter thresholds.
         * @param saturation_threshold HSV saturation threshold.
         * @param value_threshold HSV value threshold.
         * @param min_blob_area Minimum blob area in pixels.
         * @param max_blob_area Maximum blob area in pixels.
         * @maixpy maix.mygo_target_tracking.Recognizer.set_basic_thresholds
         */
        void set_basic_thresholds(int saturation_threshold,
                                  int value_threshold,
                                  int min_blob_area,
                                  int max_blob_area)
        {
            TrackerConfig cfg = tracker_.get_config();
            cfg.saturation_threshold = saturation_threshold;
            cfg.value_threshold = value_threshold;
            cfg.min_blob_area = min_blob_area;
            cfg.max_blob_area = max_blob_area;
            tracker_.set_config(cfg);
        }

        /**
         * @brief Enable Maix accelerated blob extraction with LAB thresholds.
         *
         * Threshold format follows maix.image.Image.find_blobs, each element is
         * [l_min, l_max, a_min, a_max, b_min, b_max].
         * @param lab_thresholds LAB threshold list.
         * @param x_stride X sampling stride.
         * @param y_stride Y sampling stride.
         * @maixpy maix.mygo_target_tracking.Recognizer.set_maix_blob_thresholds
         */
        void set_maix_blob_thresholds(const std::vector<std::vector<int>> &lab_thresholds,
                                      int x_stride = 2,
                                      int y_stride = 1)
        {
            TrackerConfig cfg = tracker_.get_config();
            cfg.use_maix_find_blobs = true;
            cfg.lab_thresholds = lab_thresholds;
            cfg.x_stride = x_stride;
            cfg.y_stride = y_stride;
            tracker_.set_config(cfg);
        }

        /**
         * @brief Disable Maix accelerated blob extraction and use OpenCV path.
         * @maixpy maix.mygo_target_tracking.Recognizer.disable_maix_blob
         */
        void disable_maix_blob()
        {
            TrackerConfig cfg = tracker_.get_config();
            cfg.use_maix_find_blobs = false;
            tracker_.set_config(cfg);
        }

        /**
         * @brief Detect target from an image.
         * @param image Input image.
         * @param fx Camera focal length x, <=0 means auto by width*0.6.
         * @param fy Camera focal length y, <=0 means auto by height*0.6.
         * @param cx Camera principal point x, <0 means image center.
         * @param cy Camera principal point y, <0 means image center.
         * @return DetectResult result.
         * @maixpy maix.mygo_target_tracking.Recognizer.detect
         */
        DetectResult detect(maix::image::Image &image,
                            float fx = -1.0f,
                            float fy = -1.0f,
                            float cx = -1.0f,
                            float cy = -1.0f)
        {
#ifdef MYGO_TARGETTRACKING_USE_MAIX
            TargetInfo info = tracker_.process_frame(image);
#else
            cv::Mat frame;
            maix::image::image2cv(image, frame, true, true);
            TargetInfo info = tracker_.process_frame(frame);
#endif

            DetectResult out;
            out.found = info.found;
            out.board_cx = info.board_center.x;
            out.board_cy = info.board_center.y;
            out.target_cx = info.target_center.x;
            out.target_cy = info.target_center.y;
            out.distance = info.distance;
            out.angle = info.angle;

            if (info.found) {
                const float width = static_cast<float>(image.width());
                const float height = static_cast<float>(image.height());
                const float fx_use = fx > 0.0f ? fx : width * 0.6f;
                const float fy_use = fy > 0.0f ? fy : height * 0.6f;
                const float cx_use = cx >= 0.0f ? cx : width * 0.5f;
                const float cy_use = cy >= 0.0f ? cy : height * 0.5f;
                const float dx = info.target_center.x - cx_use;
                const float dy = info.target_center.y - cy_use;
                out.pitch_error = std::atan2(dy, fy_use) * 180.0f / static_cast<float>(CV_PI);
                out.yaw_error = -std::atan2(dx, fx_use) * 180.0f / static_cast<float>(CV_PI);
            }

            return out;
        }

        /**
         * @brief Access current tracker config.
         * @return TrackerConfig object.
         * @maixpy maix.mygo_target_tracking.Recognizer.get_config
         */
        TrackerConfig get_config() const
        {
            return tracker_.get_config();
        }

        /**
         * @brief Set tracker config directly.
         * @param config TrackerConfig object.
         * @maixpy maix.mygo_target_tracking.Recognizer.set_config
         */
        void set_config(const TrackerConfig &config)
        {
            tracker_.set_config(config);
        }

    private:
        TargetTracker tracker_;
    };

    /**
     * @brief Pipeline result for tracking+control workflow.
     * @maixpy maix.mygo_target_tracking.PipelineResult
     */
    class PipelineResult
    {
    public:
        /**
         * @brief Tracking state id.
         * 0: Waiting, 1: Searching, 2: Locked, 3: Tracking.
         * @maixpy maix.mygo_target_tracking.PipelineResult.state
         */
        int state = 0;

        /**
         * @brief Current gimbal command string.
         * @maixpy maix.mygo_target_tracking.PipelineResult.command
         */
        std::string command;

        /**
         * @brief Current pitch angle in degrees.
         * @maixpy maix.mygo_target_tracking.PipelineResult.pitch_angle
         */
        float pitch_angle = 0.0f;

        /**
         * @brief Current yaw angle in degrees.
         * @maixpy maix.mygo_target_tracking.PipelineResult.yaw_angle
         */
        float yaw_angle = 0.0f;

        /**
         * @brief Current pitch speed in deg/s.
         * @maixpy maix.mygo_target_tracking.PipelineResult.pitch_speed
         */
        float pitch_speed = 0.0f;

        /**
         * @brief Current yaw speed in deg/s.
         * @maixpy maix.mygo_target_tracking.PipelineResult.yaw_speed
         */
        float yaw_speed = 0.0f;

        /**
         * @brief Whether target is found in this frame.
         * @maixpy maix.mygo_target_tracking.PipelineResult.target_found
         */
        bool target_found = false;

        /**
         * @brief Target x in pixels.
         * @maixpy maix.mygo_target_tracking.PipelineResult.target_x
         */
        float target_x = -1.0f;

        /**
         * @brief Target y in pixels.
         * @maixpy maix.mygo_target_tracking.PipelineResult.target_y
         */
        float target_y = -1.0f;

        /**
         * @brief Lock frame count.
         * @maixpy maix.mygo_target_tracking.PipelineResult.lock_count
         */
        int lock_count = 0;

        /**
         * @brief Lost frame count.
         * @maixpy maix.mygo_target_tracking.PipelineResult.lost_count
         */
        int lost_count = 0;
    };

    /**
     * @brief Full tracking pipeline wrapper for MaixPy.
     *
     * This controller exposes the original state machine + PID + gimbal command generation.
     * @maixpy maix.mygo_target_tracking.PipelineController
     */
    class PipelineController
    {
    public:
        /**
         * @brief Construct pipeline controller.
         * @maixpy maix.mygo_target_tracking.PipelineController.__init__
         */
        PipelineController() {}

        /**
         * @brief Reset pipeline state.
         * @maixpy maix.mygo_target_tracking.PipelineController.reset
         */
        void reset()
        {
            pipeline_.reset();
        }

        /**
         * @brief Set camera intrinsics used for pixel-to-angle mapping.
         * @param fx focal length x.
         * @param fy focal length y.
         * @param cx principal point x.
         * @param cy principal point y.
         * @maixpy maix.mygo_target_tracking.PipelineController.set_camera_intrinsics
         */
        void set_camera_intrinsics(float fx, float fy, float cx, float cy)
        {
            PipelineConfig cfg = pipeline_.get_config();
            cfg.fx = fx;
            cfg.fy = fy;
            cfg.cx = cx;
            cfg.cy = cy;
            pipeline_.set_config(cfg);
        }

        /**
         * @brief Set gimbal home and PID limits.
         * @param pitch_home pitch home angle in degrees.
         * @param yaw_home yaw home angle in degrees.
         * @param max_speed maximum speed in deg/s.
         * @param integral_limit PID integral limit.
         * @maixpy maix.mygo_target_tracking.PipelineController.set_control_limits
         */
        void set_control_limits(float pitch_home,
                                float yaw_home,
                                float max_speed,
                                float integral_limit)
        {
            PipelineConfig cfg = pipeline_.get_config();
            cfg.pitch_home = pitch_home;
            cfg.yaw_home = yaw_home;
            cfg.max_speed = max_speed;
            cfg.integral_limit = integral_limit;
            pipeline_.set_config(cfg);
        }

        /**
         * @brief Enable or disable serial output.
         * @param enable true to enable serial output.
         * @param device serial device path, e.g. /dev/ttyUSB0.
         * @param baud serial baudrate.
         * @maixpy maix.mygo_target_tracking.PipelineController.set_serial
         */
        void set_serial(bool enable, const std::string &device = "/dev/ttyUSB0", int baud = 115200)
        {
            PipelineConfig cfg = pipeline_.get_config();
            cfg.enable_serial = enable;
            cfg.serial_device = device;
            cfg.serial_baud = baud;
            pipeline_.set_config(cfg);
        }

        /**
         * @brief Open serial port if serial is enabled in config.
         * @return true if opened successfully.
         * @maixpy maix.mygo_target_tracking.PipelineController.open_serial
         */
        bool open_serial()
        {
            return pipeline_.open_serial();
        }

        /**
         * @brief Close serial port.
         * @maixpy maix.mygo_target_tracking.PipelineController.close_serial
         */
        void close_serial()
        {
            pipeline_.close_serial();
        }

        /**
         * @brief Handle one key command. Same behavior as original pipeline.
         * @param key key code, e.g. 32(space).
         * @maixpy maix.mygo_target_tracking.PipelineController.handle_key
         */
        void handle_key(int key)
        {
            pipeline_.handle_key(key);
        }

        /**
         * @brief Process one frame through full tracking pipeline.
         * @param image input image.
         * @param dt frame delta time in seconds.
         * @return PipelineResult lightweight result object.
         * @maixpy maix.mygo_target_tracking.PipelineController.process
         */
        PipelineResult process(maix::image::Image &image, float dt)
        {
#ifdef MYGO_TARGETTRACKING_USE_MAIX
            PipelineOutput output = pipeline_.process_frame(image, dt);
#else
            cv::Mat frame;
            maix::image::image2cv(image, frame, true, true);
            PipelineOutput output = pipeline_.process_frame(frame, dt);
#endif

            PipelineResult out;
            out.state = static_cast<int>(output.state);
            out.command = output.command;
            out.pitch_angle = output.pitch_angle;
            out.yaw_angle = output.yaw_angle;
            out.pitch_speed = output.pitch_speed;
            out.yaw_speed = output.yaw_speed;
            out.target_found = output.target_found;
            out.target_x = output.target_pos.x;
            out.target_y = output.target_pos.y;
            out.lock_count = output.lock_count;
            out.lost_count = output.lost_count;
            return out;
        }

        /**
         * @brief Get current pitch angle in degrees.
         * @return pitch angle.
         * @maixpy maix.mygo_target_tracking.PipelineController.get_pitch_angle
         */
        float get_pitch_angle() const
        {
            return pipeline_.get_pitch_angle();
        }

        /**
         * @brief Get current yaw angle in degrees.
         * @return yaw angle.
         * @maixpy maix.mygo_target_tracking.PipelineController.get_yaw_angle
         */
        float get_yaw_angle() const
        {
            return pipeline_.get_yaw_angle();
        }

        /**
         * @brief Access underlying tracker config used by pipeline.
         * @return TrackerConfig.
         * @maixpy maix.mygo_target_tracking.PipelineController.get_tracker_config
         */
        TrackerConfig get_tracker_config() const
        {
            return pipeline_.get_tracker_config();
        }

        /**
         * @brief Update underlying tracker config used by pipeline.
         * @param config TrackerConfig object.
         * @maixpy maix.mygo_target_tracking.PipelineController.set_tracker_config
         */
        void set_tracker_config(const TrackerConfig &config)
        {
            pipeline_.set_tracker_config(config);
        }

    private:
        TargetTrackingPipeline pipeline_;
    };
}
