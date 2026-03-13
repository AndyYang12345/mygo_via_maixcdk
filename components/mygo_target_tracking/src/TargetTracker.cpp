#include "TargetTracking/TargetTracker.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>
#include <opencv2/video/tracking.hpp>

#ifdef MYGO_TARGETTRACKING_USE_MAIX
#include "maix_image_cv.hpp"
#include "maix_log.hpp"
#endif

using namespace cv;
using namespace std;

// ============ 构造函数和配置管理 ============

TargetTracker::TargetTracker() 
    : frames_processed_(0), 
      successful_tracks_(0),
      frame_size_(cv::Size(640, 480)) {
    // 默认配置
    config_ = TrackerConfig();
        reset_roi_tracking();
}

void TargetTracker::set_config(const TrackerConfig& config) {
    config_ = config;
}

TrackerConfig TargetTracker::get_config() const {
    return config_;
}

// ============ 主处理流程 ============

TargetInfo TargetTracker::process_frame(const Mat& frame) {
    TargetInfo result;
    result.found = false;
    result.laser_found = false;
    frames_processed_++;
    
    // 更新帧大小
    frame_size_ = frame.size();
    
    if (frame.empty()) {
        if (config_.print_debug_info) {
            cerr << "[ERROR] Empty frame received!" << endl;
        }
        return result;
    }
    
    if (config_.print_debug_info) {
        cout << "\n=== Processing Frame #" << frames_processed_ << " ===" << endl;
        cout << "Frame size: " << frame.cols << "x" << frame.rows << endl;
    }

    bool target_ready = false;
    if (roi_tracking_active_) {
        if (update_roi_tracking(frame, result)) {
            target_ready = result.found;
        } else {
            roi_tracking_active_ = false;
        }
    }

    if (!target_ready) {
    
        // Step 1: 提取所有色块（不进行颜色匹配过滤！）
        Mat debug_mask;
        vector<ColorBlob> blobs = extract_color_blobs(frame, debug_mask);
    
        if (config_.print_debug_info) {
            cout << "Found " << blobs.size() << " color blobs" << endl;
        }
    
        if (blobs.size() < 6) {
            if (config_.print_debug_info) {
                cout << "Insufficient blobs (" << blobs.size() << "), need at least 6" << endl;
            }
            return result;
        }
    
        // Step 2: 找到中心色块
        ColorBlob* center_blob = find_center_blob(blobs);
        if (center_blob == nullptr) {
            if (config_.print_debug_info) {
                cout << "No valid center blob found!" << endl;
            }
            return result;
        }
    
           if (config_.print_debug_info) {
              cout << "Center blob found at (" << center_blob->center.x
                  << ", " << center_blob->center.y << ")" << endl;
              cout << "Center color BGR: [" << center_blob->mean_color_bgr[0]
                  << ", " << center_blob->mean_color_bgr[1]
                  << ", " << center_blob->mean_color_bgr[2] << "]" << endl;
              cout << "Center color HSV: [" << center_blob->mean_color_hsv[0]
                  << ", " << center_blob->mean_color_hsv[1]
                  << ", " << center_blob->mean_color_hsv[2] << "]" << endl;
              cout << "Center is dark: " << (center_blob->is_dark ? "YES" : "NO") << endl;
           }
    
        // Step 3: 找到匹配的目标色块
        ColorBlob* target_blob = find_matching_target(blobs, *center_blob);
        if (target_blob == nullptr) {
            if (config_.print_debug_info) {
                cout << "No matching target blob found!" << endl;
            }
            return result;
        }
    
        // Step 4: 计算结果
        result.found = true;
        result.board_center = center_blob->center;
        result.target_center = target_blob->center;
        last_board_position_ = result.board_center;
    
        Point2f delta = target_blob->center - center_blob->center;
        result.distance = norm(delta);
        result.angle = atan2(delta.y, delta.x) * 180.0f / CV_PI;
    
        successful_tracks_++;
        if (config_.print_debug_info) {
            cout << "SUCCESS: Target found!" << endl;
            cout << "  Target position: (" << target_blob->center.x
                 << ", " << target_blob->center.y << ")" << endl;
            cout << "  Distance: " << result.distance << " pixels" << endl;
            cout << "  Angle: " << result.angle << " degrees" << endl;
        }
    
        init_roi_tracking(frame, *target_blob);

        if (config_.show_debug_windows) {
            Mat debug_frame = frame.clone();
            draw_debug_info(debug_frame, blobs, center_blob, target_blob);

            vector<Mat> debug_images;
            debug_images.push_back(debug_frame);
            debug_images.push_back(debug_mask);

            Mat combined;
            hconcat(debug_images, combined);

            resize(combined, combined, Size(), 0.5, 0.5);
            imshow("Target Tracker Debug", combined);
            waitKey(1);
        }
    }

    if (result.found && config_.enable_laser_detection) {
        cv::Point2f laser_center;
        cv::Mat laser_mask;
        result.laser_found = detect_laser_dot(frame, result.target_center, laser_center, &laser_mask);
        if (result.laser_found) {
            result.laser_center = laser_center;
            result.laser_to_target_distance = cv::norm(result.target_center - result.laser_center);
            has_previous_laser_ = true;
            last_laser_position_ = laser_center;
        } else {
            result.laser_center = cv::Point2f(-1.0f, -1.0f);
            result.laser_to_target_distance = 0.0f;
        }

        if (config_.show_debug_windows) {
            cv::Mat laser_vis;
            cv::cvtColor(laser_mask, laser_vis, cv::COLOR_GRAY2BGR);
            if (result.laser_found) {
                cv::circle(laser_vis, result.laser_center, 5, cv::Scalar(0, 255, 255), 2);
            }
            imshow("Laser Dot Mask", laser_vis);
            waitKey(1);
        }
    }

    return result;
}

#ifdef MYGO_TARGETTRACKING_USE_MAIX
TargetInfo TargetTracker::process_frame(maix::image::Image& image) {
    cv::Mat mat;
    if (maix::image::image2cv(image, mat, true, true) != maix::err::ERR_NONE) {
        TargetInfo result;
        result.found = false;
        return result;
    }
    return process_frame(mat);
}
#endif

// ============ ROI 跟踪 ============

void TargetTracker::reset_roi_tracking() {
    roi_tracking_active_ = false;
    has_previous_target_ = false;
    kalman_initialized_ = false;
    has_previous_laser_ = false;
    last_target_position_ = cv::Point2f(-1.0f, -1.0f);
    last_board_position_ = cv::Point2f(-1.0f, -1.0f);
    last_laser_position_ = cv::Point2f(-1.0f, -1.0f);
}

bool TargetTracker::detect_laser_dot(const cv::Mat& frame,
                                     const cv::Point2f& target_hint,
                                     cv::Point2f& out_center,
                                     cv::Mat* out_mask) {
    if (frame.empty()) {
        return false;
    }

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask1, mask2;
    cv::inRange(hsv,
                cv::Scalar(config_.laser_hue_low_1, config_.laser_min_saturation, config_.laser_min_value),
                cv::Scalar(config_.laser_hue_high_1, 255, 255),
                mask1);
    cv::inRange(hsv,
                cv::Scalar(config_.laser_hue_low_2, config_.laser_min_saturation, config_.laser_min_value),
                cv::Scalar(config_.laser_hue_high_2, 255, 255),
                mask2);

    cv::Mat mask = mask1 | mask2;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);

    if (out_mask) {
        *out_mask = mask;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    bool best_found = false;
    float best_score = -std::numeric_limits<float>::max();
    cv::Point2f best_center(-1.0f, -1.0f);

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < config_.laser_min_blob_area || area > config_.laser_max_blob_area) {
            continue;
        }

        const cv::Moments m = cv::moments(contour);
        if (m.m00 <= 1e-5) {
            continue;
        }

        const cv::Point2f center(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
        const double perimeter = cv::arcLength(contour, true);
        const float circularity = (perimeter > 1e-4)
                                      ? static_cast<float>((4.0 * CV_PI * area) / (perimeter * perimeter))
                                      : 0.0f;

        cv::Rect br = cv::boundingRect(contour);
        br &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (br.width <= 0 || br.height <= 0) {
            continue;
        }
        cv::Mat hsv_roi = hsv(br);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_roi, hsv_channels);
        double max_v = 0.0;
        cv::minMaxLoc(hsv_channels[2], nullptr, &max_v);

        float score = 0.0f;
        if (target_hint.x >= 0.0f && target_hint.y >= 0.0f) {
            const float dist_to_target = cv::norm(center - target_hint);
            score += 1.6f * std::max(0.0f, 1.0f - dist_to_target / std::max(1, config_.laser_search_radius));
        }
        if (has_previous_laser_) {
            const float dist_to_last = cv::norm(center - last_laser_position_);
            score += std::max(0.0f, 1.0f - dist_to_last / std::max(1, config_.laser_search_radius));
        }
        score += config_.laser_circularity_weight * circularity;
        score += 0.2f * static_cast<float>(area / std::max(1, config_.laser_max_blob_area));
        score += 0.35f * static_cast<float>(max_v / 255.0);

        if (score > best_score) {
            best_score = score;
            best_center = center;
            best_found = true;
        }
    }

    if (!best_found) {
        return false;
    }

    out_center = best_center;
    return true;
}

bool TargetTracker::init_roi_tracking(const cv::Mat& frame, const ColorBlob& target_blob) {
    if (frame.empty()) {
        return false;
    }

    target_color_bgr_ = target_blob.mean_color_bgr;
    target_color_hsv_ = bgr_to_hsv(target_color_bgr_);
    last_target_position_ = target_blob.center;
    has_previous_target_ = true;
    roi_tracking_active_ = true;

    if (config_.use_kalman) {
        kalman_ = cv::KalmanFilter(4, 2, 0, CV_32F);
        kalman_.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, config_.kalman_dt, 0,
            0, 1, 0, config_.kalman_dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kalman_.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);
        setIdentity(kalman_.processNoiseCov, cv::Scalar(1e-2));
        setIdentity(kalman_.measurementNoiseCov, cv::Scalar(1e-1));
        setIdentity(kalman_.errorCovPost, cv::Scalar(1));
        kalman_.statePost = (cv::Mat_<float>(4, 1) <<
            target_blob.center.x, target_blob.center.y, 0.0f, 0.0f);
        kalman_initialized_ = true;
    }

    return true;
}

bool TargetTracker::update_roi_tracking(const cv::Mat& frame, TargetInfo& result) {
    if (!has_previous_target_) {
        return false;
    }

    cv::Point2f predict_point = last_target_position_;
    if (config_.use_kalman && kalman_initialized_) {
        cv::Mat prediction = kalman_.predict();
        predict_point = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
    }

    cv::Rect roi;
    roi.x = std::max(0, static_cast<int>(predict_point.x) - config_.roi_padding);
    roi.y = std::max(0, static_cast<int>(predict_point.y) - config_.roi_padding);
    roi.width = std::min(frame.cols - roi.x, config_.roi_padding * 2);
    roi.height = std::min(frame.rows - roi.y, config_.roi_padding * 2);

    if (roi.width <= 0 || roi.height <= 0) {
        return false;
    }

    cv::Point2f detected_center;
    bool detected = detect_target_in_roi(frame, roi, detected_center);

    if (detected) {
        last_target_position_ = detected_center;
        if (config_.use_kalman && kalman_initialized_) {
            cv::Mat measurement = (cv::Mat_<float>(2, 1) << detected_center.x, detected_center.y);
            kalman_.correct(measurement);
        }
    } else if (config_.use_kalman && kalman_initialized_) {
        last_target_position_ = predict_point;
    } else {
        return false;
    }

    result.found = detected;
    result.target_center = last_target_position_;
    result.board_center = last_board_position_;
    if (result.board_center.x >= 0.0f) {
        cv::Point2f delta = result.target_center - result.board_center;
        result.distance = norm(delta);
        result.angle = atan2(delta.y, delta.x) * 180.0f / CV_PI;
    }

    return detected;
}

bool TargetTracker::detect_target_in_roi(const cv::Mat& frame, const cv::Rect& roi,
                                         cv::Point2f& out_center) const {
    cv::Mat roi_bgr = frame(roi);
    cv::Mat roi_hsv;
    cv::cvtColor(roi_bgr, roi_hsv, cv::COLOR_BGR2HSV);

    int hue = static_cast<int>(target_color_hsv_[0]);
    int sat = static_cast<int>(target_color_hsv_[1]);
    int val = static_cast<int>(target_color_hsv_[2]);

    int h_low = hue - config_.roi_hue_threshold;
    int h_high = hue + config_.roi_hue_threshold;
    int s_low = std::max(0, sat - config_.roi_sat_threshold);
    int s_high = std::min(255, sat + config_.roi_sat_threshold);
    int v_low = std::max(0, val - config_.roi_val_threshold);
    int v_high = std::min(255, val + config_.roi_val_threshold);

    cv::Mat mask, mask1, mask2;
    if (h_low < 0 || h_high > 179) {
        int h_low_wrap = (h_low + 180) % 180;
        int h_high_wrap = h_high % 180;
        cv::inRange(roi_hsv, cv::Scalar(0, s_low, v_low), cv::Scalar(h_high_wrap, s_high, v_high), mask1);
        cv::inRange(roi_hsv, cv::Scalar(h_low_wrap, s_low, v_low), cv::Scalar(179, s_high, v_high), mask2);
        mask = mask1 | mask2;
    } else {
        cv::inRange(roi_hsv, cv::Scalar(h_low, s_low, v_low), cv::Scalar(h_high, s_high, v_high), mask);
    }

    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_area = 0.0;
    cv::Point2f best_center;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < config_.roi_min_blob_area || area > config_.roi_max_blob_area) {
            continue;
        }
        cv::Moments m = cv::moments(contour);
        if (m.m00 <= 0) {
            continue;
        }
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
        if (area > best_area) {
            best_area = area;
            best_center = center;
        }
    }

    if (best_area <= 0.0) {
        return false;
    }

    out_center = cv::Point2f(best_center.x + roi.x, best_center.y + roi.y);
    return true;
}

// ============ 核心处理函数 ============

vector<ColorBlob> TargetTracker::extract_color_blobs(const Mat& frame, Mat& debug_mask) {
    vector<ColorBlob> blobs;

#ifdef MYGO_TARGETTRACKING_USE_MAIX
    if (config_.use_maix_find_blobs) {
        maix::image::Image *img = maix::image::cv2image(const_cast<cv::Mat&>(frame), true, true);
        if (!img) {
            debug_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            return blobs;
        }
        blobs = extract_color_blobs(*img);
        delete img;
        debug_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        return blobs;
    }
#endif
    
    if (config_.print_debug_info) {
        cout << "[DEBUG] extract_color_blobs: Starting..." << endl;
    }
    
    // 1. 预处理：高斯模糊
    Mat blurred;
    GaussianBlur(frame, blurred, 
                 Size(config_.blur_size, config_.blur_size), 
                 config_.blur_sigma);
    
    // 2. 转换到HSV颜色空间
    Mat hsv;
    cvtColor(blurred, hsv, COLOR_BGR2HSV);
    
    // 3. 分离通道
    vector<Mat> channels;
    split(hsv, channels);
    Mat hue = channels[0];
    Mat saturation = channels[1];
    Mat value = channels[2];
    
    // 分析图像整体亮度
    Scalar mean_value = mean(value);
    bool image_is_bright = mean_value[0] > 200;
    
    if (config_.print_debug_info) {
        cout << "[DEBUG] Image mean brightness: " << mean_value[0] << endl;
        cout << "[DEBUG] Image is bright: " << (image_is_bright ? "YES" : "NO") << endl;
    }
    
    // 4. 创建掩码
    Mat sat_mask, val_mask, combined_mask;
    
    // 饱和度掩码
    threshold(saturation, sat_mask, config_.saturation_threshold, 255, THRESH_BINARY);
    
    // 自适应亮度阈值
    int adaptive_val_thresh;
    if (image_is_bright) {
        adaptive_val_thresh = max(config_.value_threshold, 180);
    } else {
        adaptive_val_thresh = config_.value_threshold;
    }
    
    threshold(value, val_mask, adaptive_val_thresh, 255, THRESH_BINARY);
    
    if (config_.print_debug_info) {
        int sat_pixels = countNonZero(sat_mask);
        int val_pixels = countNonZero(val_mask);
        cout << "[DEBUG] Using adaptive brightness threshold: " << adaptive_val_thresh << endl;
        cout << "[DEBUG] Saturation mask: " << sat_pixels << " pixels (" 
             << (sat_pixels * 100.0 / (frame.cols * frame.rows)) << "%)" << endl;
        cout << "[DEBUG] Value mask: " << val_pixels << " pixels (" 
             << (val_pixels * 100.0 / (frame.cols * frame.rows)) << "%)" << endl;
    }
    
    // 5. 组合掩码
    if (image_is_bright) {
        // 亮图像：主要使用饱和度掩码
        combined_mask = sat_mask;
        
        // 特别处理深色：添加低亮度区域
        Mat low_val_mask;
        threshold(value, low_val_mask, 80, 255, THRESH_BINARY_INV);
        Mat dark_sat_mask;
        threshold(saturation, dark_sat_mask, 80, 255, THRESH_BINARY);
        Mat dark_mask = low_val_mask & dark_sat_mask;
        combined_mask = combined_mask | dark_mask;
    } else {
        // 正常图像：同时满足饱和度和亮度条件
        combined_mask = sat_mask & val_mask;
    }
    
    // 6. 形态学操作
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(combined_mask, combined_mask, MORPH_CLOSE, kernel);
    morphologyEx(combined_mask, combined_mask, MORPH_OPEN, kernel);
    
    // 7. 查找轮廓
    vector<vector<Point>> contours;
    findContours(combined_mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (config_.print_debug_info) {
        cout << "[DEBUG] Found " << contours.size() << " contours before filtering" << endl;
    }
    
    // 8. 提取色块信息（仅基于几何特征过滤）
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        
        // 面积过滤
        if (area < config_.min_blob_area || area > config_.max_blob_area) {
            if (config_.print_debug_info && area < config_.min_blob_area) {
                cout << "[DEBUG] Contour " << i << " rejected: area " << area 
                     << " < min " << config_.min_blob_area << endl;
            }
            continue;
        }
        
        ColorBlob blob;
        blob.bounding_rect = boundingRect(contours[i]);
        blob.area = area;
        blob.circularity = calculate_circularity(contours[i]);
        
        // 圆形度过滤
        if (blob.circularity < config_.min_circularity) {
            if (config_.print_debug_info) {
                cout << "[DEBUG] Contour " << i << " rejected: circularity " << blob.circularity 
                     << " < threshold " << config_.min_circularity << endl;
            }
            continue;
        }
        
        // 计算质心
        Moments m = moments(contours[i]);
        if (m.m00 > 0) {
            blob.center = Point2f(m.m10 / m.m00, m.m01 / m.m00);
        } else {
            blob.center = Point2f(blob.bounding_rect.x + blob.bounding_rect.width / 2,
                                 blob.bounding_rect.y + blob.bounding_rect.height / 2);
        }
        
        // 计算平均颜色
        Mat roi = frame(blob.bounding_rect);
        blob.mean_color_bgr = mean(roi);
        blob.mean_color_hsv = bgr_to_hsv(blob.mean_color_bgr);
        blob.mean_color_lab = bgr_to_lab(blob.mean_color_bgr);
        
        // 判断是否为深色
        blob.is_dark = is_dark_color(blob.mean_color_bgr, config_.dark_brightness_threshold);
        
        if (config_.print_debug_info) {
            cout << "[DEBUG] Contour " << i << " accepted:" << endl;
            cout << "[DEBUG]   Position: (" << blob.center.x << ", " << blob.center.y << ")" << endl;
            cout << "[DEBUG]   Area: " << blob.area << ", Circularity: " << blob.circularity << endl;
            cout << "[DEBUG]   BGR: (" << blob.mean_color_bgr[0] << ", " 
                 << blob.mean_color_bgr[1] << ", " << blob.mean_color_bgr[2] << ")" << endl;
            cout << "[DEBUG]   HSV: (" << blob.mean_color_hsv[0] << ", " 
                 << blob.mean_color_hsv[1] << ", " << blob.mean_color_hsv[2] << ")" << endl;
            cout << "[DEBUG]   Is dark: " << (blob.is_dark ? "YES" : "NO") << endl;
        }
        
        blobs.push_back(blob);
    }
    
    // 保存调试掩码
    debug_mask = combined_mask;
    
    if (config_.print_debug_info) {
        cout << "[DEBUG] extract_color_blobs: Returning " << blobs.size() << " blobs" << endl;
    }
    
    return blobs;
}

#ifdef MYGO_TARGETTRACKING_USE_MAIX
vector<ColorBlob> TargetTracker::extract_color_blobs(maix::image::Image& image) {
    vector<ColorBlob> blobs;

    if (config_.lab_thresholds.empty()) {
        return blobs;
    }

    cv::Mat mat;
    if (maix::image::image2cv(image, mat, true, true) != maix::err::ERR_NONE) {
        return blobs;
    }

    std::vector<int> roi = {0, 0, image.width(), image.height()};
    auto maix_blobs = image.find_blobs(
        config_.lab_thresholds,
        false,
        roi,
        config_.x_stride,
        config_.y_stride,
        config_.min_blob_area,
        config_.pixels_threshold,
        config_.merge_blobs,
        config_.merge_margin
    );

    for (auto& blob_src : maix_blobs) {
        int pixels = blob_src.pixels();
        if (pixels < config_.min_blob_area || pixels > config_.max_blob_area) {
            continue;
        }

        ColorBlob blob;
        blob.bounding_rect = cv::Rect(blob_src.x(), blob_src.y(), blob_src.w(), blob_src.h());
        blob.center = cv::Point2f(blob_src.cxf(), blob_src.cyf());
        blob.area = pixels;
        blob.circularity = blob_src.roundness();
        if (blob.circularity < config_.min_circularity) {
            continue;
        }

        cv::Rect safe_rect = blob.bounding_rect & cv::Rect(0, 0, mat.cols, mat.rows);
        if (safe_rect.width <= 0 || safe_rect.height <= 0) {
            continue;
        }

        cv::Mat blob_roi = mat(safe_rect);
        blob.mean_color_bgr = cv::mean(blob_roi);
        blob.mean_color_hsv = bgr_to_hsv(blob.mean_color_bgr);
        blob.mean_color_lab = bgr_to_lab(blob.mean_color_bgr);
        blob.is_dark = is_dark_color(blob.mean_color_bgr, config_.dark_brightness_threshold);
        blobs.push_back(blob);
    }

    return blobs;
}
#endif

ColorBlob* TargetTracker::find_center_blob(vector<ColorBlob>& blobs) {
    if (blobs.empty()) return nullptr;
    
    ColorBlob* best_center = nullptr;
    float best_score = -1.0f;
    
    // 图像中心
    Point2f image_center(frame_size_.width / 2.0f, frame_size_.height / 2.0f);
    
    for (auto& blob : blobs) {
        // 计算分数：考虑圆形度和位置
        float circularity_score = blob.circularity;
        
        // 位置分数：距离图像中心越近，分数越高
        float dist_to_center = norm(blob.center - image_center);
        float max_dist = norm(Point2f(0, 0) - image_center);
        float position_score = 1.0f - (dist_to_center / max_dist);
        
        // 周围色块数量
        int surrounding_count = 0;
        for (const auto& other : blobs) {
            if (&blob == &other) continue;
            
            float dist = norm(other.center - blob.center);
            if (dist >= config_.min_distance_to_center && 
                dist <= config_.max_distance_to_center) {
                surrounding_count++;
            }
        }
        
        float surround_score = surrounding_count / 5.0f;  // 期望5个
        
        // 综合分数
        float total_score = circularity_score * 0.4f + 
                           position_score * 0.3f + 
                           surround_score * 0.3f;
        
        if (total_score > best_score) {
            best_score = total_score;
            best_center = &blob;
        }
    }
    
    if (config_.print_debug_info && best_center != nullptr) {
        cout << "Center blob score: " << best_score << endl;
        cout << "Surrounding blobs: ";
        for (const auto& blob : blobs) {
            if (&blob == best_center) continue;
            float dist = norm(blob.center - best_center->center);
            if (dist >= config_.min_distance_to_center && 
                dist <= config_.max_distance_to_center) {
                cout << "(" << blob.center.x << "," << blob.center.y << ") ";
            }
        }
        cout << endl;
    }
    
    return best_center;
}

ColorBlob* TargetTracker::find_matching_target(const vector<ColorBlob>& blobs, 
                                              const ColorBlob& center_blob) {
    // 计算中心色块是否为暗色
    bool center_is_dark = is_dark_color(center_blob.mean_color_bgr, config_.dark_brightness_threshold);
    
    if (config_.print_debug_info) {
        cout << "Finding match for center at (" << center_blob.center.x 
             << ", " << center_blob.center.y << ")" << endl;
        cout << "Center color BGR: (" << center_blob.mean_color_bgr[0] << ", "
             << center_blob.mean_color_bgr[1] << ", " << center_blob.mean_color_bgr[2] << ")" << endl;
        cout << "Center is dark: " << (center_is_dark ? "YES" : "NO") << endl;
    }
    
    // 收集所有候选色块的相似度
    vector<double> similarities;
    vector<const ColorBlob*> candidates;
    
    for (const auto& blob : blobs) {
        if (&blob == &center_blob) continue;
        if (!is_valid_surrounding_blob(blob, center_blob)) continue;
        
        float similarity = calculate_color_similarity(
            center_blob.mean_color_bgr, 
            blob.mean_color_bgr, 
            center_is_dark
        );
        
        similarities.push_back(similarity);
        candidates.push_back(&blob);
        
        if (config_.print_debug_info) {
        cout << "Raw similarities with multi-space analysis:" << endl;
        for (size_t i = 0; i < similarities.size(); ++i) {
            // 计算多空间相似度详情
            ColorSimilarity sim_detail = calculate_multi_space_similarity(
                center_blob.mean_color_bgr, 
                candidates[i]->mean_color_bgr,
                center_is_dark
            );
            
            cout << "  Candidate " << i << " at (" << candidates[i]->center.x 
                 << "," << candidates[i]->center.y << "):" << endl;
            cout << "    BGR=" << std::setprecision(3) << sim_detail.bgr_sim
                 << ", HSV=" << sim_detail.hsv_sim
                 << ", Lab=" << sim_detail.lab_sim
                 << ", Combined=" << similarities[i] << endl;
            
            // 显示颜色类型
            cv::Scalar hsv = bgr_to_hsv(candidates[i]->mean_color_bgr);
            int color_type = classify_color_type(hsv);
            cout << "    Color type: " << color_type 
                 << " (H=" << hsv[0] << ", S=" << hsv[1] << ", V=" << hsv[2] << ")" << endl;
        }
        }
    }
    
    if (candidates.empty()) {
        if (config_.print_debug_info) {
            cout << "No valid surrounding blobs!" << endl;
        }
        return nullptr;
    }
    
    // 直接使用原始相似度，不进行归一化
    // 这样可以保留原始相似度的绝对差异，对蓝色/紫色等相似颜色的区分更有利
    
    // 第一步：检查是否有色调完全相同或极其相近的候选（对蓝紫色关键）
    cv::Scalar center_hsv = bgr_to_hsv(center_blob.mean_color_bgr);
    float center_hue = center_hsv[0];
    
    const float HUE_MATCH_THRESHOLD = 8.0f;  // 色调匹配阈值（度）
    int hue_matched_idx = -1;
    float best_hue_diff = HUE_MATCH_THRESHOLD;
    
    for (size_t i = 0; i < candidates.size(); ++i) {
        cv::Scalar candidate_hsv = bgr_to_hsv(candidates[i]->mean_color_bgr);
        float candidate_hue = candidate_hsv[0];
        
        // 计算环形Hue差异
        float hue_diff = abs(candidate_hue - center_hue);
        hue_diff = min(hue_diff, 180.0f - hue_diff);
        
        if (hue_diff < best_hue_diff) {
            best_hue_diff = hue_diff;
            hue_matched_idx = i;
        }
    }
    
    // 如果找到色调匹配的候选，从这些候选中选出最高的相似度
    int best_idx = 0;
    if (hue_matched_idx != -1 && best_hue_diff < HUE_MATCH_THRESHOLD) {
        if (config_.print_debug_info) {
            cout << "[HUE-MATCH] Found candidate with Hue diff=" << best_hue_diff << " at index " << hue_matched_idx << endl;
        }
        
        // 在色调匹配的候选中选择相似度最高的
        best_idx = hue_matched_idx;
        float max_sim_in_hue_matched = similarities[hue_matched_idx];
        
        for (size_t i = 0; i < candidates.size(); ++i) {
            cv::Scalar candidate_hsv = bgr_to_hsv(candidates[i]->mean_color_bgr);
            float candidate_hue = candidate_hsv[0];
            float hue_diff = abs(candidate_hue - center_hue);
            hue_diff = min(hue_diff, 180.0f - hue_diff);
            
            if (hue_diff < HUE_MATCH_THRESHOLD && similarities[i] > max_sim_in_hue_matched) {
                max_sim_in_hue_matched = similarities[i];
                best_idx = i;
            }
        }
        
        if (config_.print_debug_info) {
            cout << "[HUE-MATCH-SELECTED] Chose candidate with best similarity among Hue-matched ones" << endl;
        }
    } else {
        // 否则按原始相似度选择
        for (size_t i = 1; i < similarities.size(); ++i) {
            if (similarities[i] > similarities[best_idx]) {
                best_idx = i;
            }
        }
        
        if (config_.print_debug_info) {
            cout << "[NO-HUE-MATCH] No Hue-matched candidate, using best overall similarity" << endl;
        }
    }
    
    if (config_.print_debug_info) {
        cout << "Raw similarities (with Hue-consistency check):" << endl;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cv::Scalar candidate_hsv = bgr_to_hsv(candidates[i]->mean_color_bgr);
            float candidate_hue = candidate_hsv[0];
            float hue_diff = abs(candidate_hue - center_hue);
            hue_diff = min(hue_diff, 180.0f - hue_diff);
            cout << "  Candidate " << i << " at (" << candidates[i]->center.x 
                 << "," << candidates[i]->center.y << "): similarity=" 
                 << similarities[i] << ", Hue_diff=" << hue_diff << endl;
        }
        cout << "Selected target at (" << candidates[best_idx]->center.x 
             << ", " << candidates[best_idx]->center.y << ")" << endl;
        cout << "Best similarity: " << similarities[best_idx] << endl;
    }
    
    return const_cast<ColorBlob*>(candidates[best_idx]);
}
// ============ 颜色匹配函数 ============

double TargetTracker::calculate_color_similarity(const cv::Scalar& color1, const cv::Scalar& color2, bool center_is_dark) {
    // 使用多颜色空间融合方法
    ColorSimilarity sim = calculate_multi_space_similarity(color1, color2, center_is_dark);
    return sim.combined_sim;
}
// 在TargetTracker.cpp中实现

// 颜色类型分类
int TargetTracker::classify_color_type(const cv::Scalar& hsv) {
    float hue = hsv[0];      // 0-180
    float saturation = hsv[1]; // 0-255
    float value = hsv[2];      // 0-255
    
    // 根据HSV值分类颜色类型
    if (value < 50) return 0;  // 黑色/非常暗
    
    if (saturation < 50) {
        if (value > 200) return 1;  // 白色/非常亮
        return 2;  // 灰色/中性色
    }
    
    // 彩色分类
    if (hue >= 0 && hue < 15) return 3;    // 红色
    else if (hue >= 15 && hue < 45) return 4;   // 橙色
    else if (hue >= 45 && hue < 75) return 5;   // 黄色
    else if (hue >= 75 && hue < 105) return 6;  // 黄绿色
    else if (hue >= 105 && hue < 135) return 7; // 绿色
    else if (hue >= 135 && hue < 165) return 8; // 青色
    else if (hue >= 165 && hue < 195) return 9; // 蓝色
    else if (hue >= 195 && hue < 225) return 10; // 紫色
    else if (hue >= 225 && hue < 255) return 11; // 粉色
    else return 12; // 红色（环状）
}

TargetTracker::ColorSimilarity TargetTracker::calculate_multi_space_similarity(const cv::Scalar& color1, const cv::Scalar& color2, bool center_is_dark) {
    ColorSimilarity result;
    
    // 转换为各种颜色空间
    cv::Scalar hsv1 = bgr_to_hsv(color1);
    cv::Scalar hsv2 = bgr_to_hsv(color2);
    cv::Scalar lab1 = bgr_to_lab(color1);
    cv::Scalar lab2 = bgr_to_lab(color2);
    
    // 1. BGR相似度（适合所有颜色）
    double bgr_db = color1[0] - color2[0];
    double bgr_dg = color1[1] - color2[1];
    double bgr_dr = color1[2] - color2[2];
    double bgr_dist = sqrt(bgr_db*bgr_db + bgr_dg*bgr_dg + bgr_dr*bgr_dr);
    result.bgr_sim = exp(-bgr_dist / 100.0);
    
    // 2. HSV相似度（特别适合区分色调）
    double hue_diff = abs(hsv1[0] - hsv2[0]);
    hue_diff = min(hue_diff, 180.0 - hue_diff);  // 环形处理
    
    double sat_diff = abs(hsv1[1] - hsv2[1]);
    double val_diff = abs(hsv1[2] - hsv2[2]);
    
    // 加权HSV相似度
    double hsv_dist = hue_diff * 2.0 + sat_diff * 0.3 + val_diff * 0.2;
    result.hsv_sim = exp(-hsv_dist / 100.0);
    
    // 3. Lab相似度（考虑人眼感知）
    double lab_L_diff = lab1[0] - lab2[0];
    double lab_a_diff = lab1[1] - lab2[1];
    double lab_b_diff = lab1[2] - lab2[2];
    double lab_dist = sqrt(lab_L_diff*lab_L_diff + lab_a_diff*lab_a_diff + lab_b_diff*lab_b_diff);
    result.lab_sim = 1.0 - min(lab_dist / 150.0, 1.0);  // Lab距离通常0-100+
    
    // 4. 颜色类型感知的加权组合
    int color_type1 = classify_color_type(hsv1);
    
    // 根据颜色类型调整权重
    double w_bgr = 0.3, w_hsv = 0.4, w_lab = 0.3;  // 默认权重
    
    // 特定颜色类型的权重调整
    if (color_type1 >= 9 && color_type1 <= 11) {  // 蓝色、紫色、粉色
        // 对这些颜色，增加HSV权重（色调区分重要）
        w_hsv = 0.6;
        w_bgr = 0.2;
        w_lab = 0.2;
    } else if (center_is_dark) {
        // 暗色：增加BGR权重
        w_bgr = 0.5;
        w_hsv = 0.3;
        w_lab = 0.2;
    } else if (hsv1[1] < 100) {  // 低饱和度
        // 低饱和度颜色：增加Lab权重
        w_lab = 0.5;
        w_bgr = 0.3;
        w_hsv = 0.2;
    }
    
    result.combined_sim = w_bgr * result.bgr_sim + 
                          w_hsv * result.hsv_sim + 
                          w_lab * result.lab_sim;
    
    return result;
}

// ============ 辅助函数 ============

// 在TargetTracker.cpp中实现
double TargetTracker::calculate_circularity(const vector<Point>& contour) {
    double area = contourArea(contour);
    double perimeter = arcLength(contour, true);
    
    if (perimeter == 0) return 0;
    
    double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
    return circularity;
}

bool TargetTracker::is_valid_surrounding_blob(const ColorBlob& blob, 
                                             const ColorBlob& center) {
    // 检查距离
    float distance = norm(blob.center - center.center);
    
    if (distance < config_.min_distance_to_center || 
        distance > config_.max_distance_to_center) {
        return false;
    }
    
    return true;
}

Scalar TargetTracker::bgr_to_hsv(const Scalar& bgr) {
    Mat bgr_mat(1, 1, CV_8UC3, Scalar(bgr[0], bgr[1], bgr[2]));
    Mat hsv_mat;
    cvtColor(bgr_mat, hsv_mat, COLOR_BGR2HSV);
    Vec3b hsv = hsv_mat.at<Vec3b>(0, 0);
    return Scalar(hsv[0], hsv[1], hsv[2]);
}
Scalar TargetTracker::bgr_to_lab(const Scalar& bgr) {
    // 创建一个1x1的BGR图像
    Mat bgr_mat(1, 1, CV_8UC3, Scalar(bgr[0], bgr[1], bgr[2]));
    Mat lab_mat;
    
    // 将BGR转换为Lab颜色空间
    cvtColor(bgr_mat, lab_mat, COLOR_BGR2Lab);
    
    // 提取Lab值
    Vec3b lab_values = lab_mat.at<Vec3b>(0, 0);

    // 在Lab颜色空间中：
    // L: 0-100 (亮度) -> 映射到0-255
    // a: -127 to 128 (绿到红) -> 映射到0-255
    // b: -127 to 128 (蓝到黄) -> 映射到0-255
    
    return Scalar(lab_values[0], lab_values[1], lab_values[2]);
}

bool TargetTracker::is_dark_color(const Scalar& bgr, int threshold) {
    // 计算亮度: 0.299*R + 0.587*G + 0.114*B
    float brightness = 0.299f * bgr[2] + 0.587f * bgr[1] + 0.114f * bgr[0];
    return brightness < threshold;
}

// ============ 调试功能 ============

void TargetTracker::draw_debug_info(Mat& frame, 
                                   const vector<ColorBlob>& blobs,
                                   const ColorBlob* center,
                                   const ColorBlob* target) {
    // 绘制所有色块
    for (const auto& blob : blobs) {
        // 边界框
        rectangle(frame, blob.bounding_rect, Scalar(255, 0, 0), 2);
        
        // 中心点
        circle(frame, blob.center, 3, Scalar(0, 255, 0), -1);
        
        // 面积和圆形度
        string info = format("A:%.0f C:%.2f", blob.area, blob.circularity);
        putText(frame, info, 
                Point(blob.bounding_rect.x, blob.bounding_rect.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }
    
    // 绘制中心色块
    if (center != nullptr) {
        circle(frame, center->center, 8, Scalar(0, 255, 255), 3);
        putText(frame, "CENTER", 
                Point(center->center.x + 10, center->center.y),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
    }
    
    // 绘制目标色块
    if (target != nullptr && center != nullptr) {
        // 目标色块
        circle(frame, target->center, 8, Scalar(0, 0, 255), 3);
        
        // 连接线
        line(frame, center->center, target->center, Scalar(0, 255, 0), 2);
        
        // 距离和角度信息
        float distance = norm(target->center - center->center);
        float angle = atan2(target->center.y - center->center.y,
                           target->center.x - center->center.x) * 180 / CV_PI;
        
        string info = format("Dist: %.1fpx, Angle: %.1f deg", distance, angle);
        putText(frame, info, Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        
        // 显示颜色信息
        string color_info = format("Center: (%.0f,%.0f,%.0f)", 
                                  center->mean_color_bgr[0],
                                  center->mean_color_bgr[1],
                                  center->mean_color_bgr[2]);
        putText(frame, color_info, Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
    }
}


void TargetTracker::enable_debug(bool enabled) {
    config_.show_debug_windows = enabled;
    config_.print_debug_info = enabled;
}

void TargetTracker::reset_statistics() {
    frames_processed_ = 0;
    successful_tracks_ = 0;
}

void TargetTracker::print_statistics() const {
    cout << "\n=== Tracker Statistics ===" << endl;
    cout << "Frames processed: " << frames_processed_ << endl;
    cout << "Successful tracks: " << successful_tracks_ << endl;
    
    if (frames_processed_ > 0) {
        float success_rate = (float)successful_tracks_ / frames_processed_ * 100;
        cout << "Success rate: " << success_rate << "%" << endl;
    }
    
    cout << "=========================" << endl;
}