#ifndef TARGET_TRACKER_HPP
#define TARGET_TRACKER_HPP

#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <string>

// 配置结构体
struct TrackerConfig {
    // 预处理参数
    int blur_size = 5;
    double blur_sigma = 1.5;
    
    // 掩码参数
    int saturation_threshold = 50;
    int value_threshold = 50;
    int dark_brightness_threshold = 100;
    
    // 色块过滤参数
    int min_blob_area = 500;
    int max_blob_area = 10000;
    float min_circularity = 0.6f;
    
    // 中心检测参数
    float min_distance_to_center = 50.0f;
    float max_distance_to_center = 250.0f;
    
    // 颜色匹配参数
    float color_similarity_threshold = 0.6f;  // 颜色相似度阈值
    float bgr_distance_threshold = 150.0f;    // BGR距离阈值
    float hue_similarity_threshold = 30.0f;   // 色调相似度阈值
    
    // 新增：归一化匹配参数
    bool use_softmax_normalization = true;      // 是否使用softmax归一化
    float min_confidence_ratio = 1.5f;          // 最小置信度比值（最高/第二高）
    float min_absolute_similarity = 0.2f;       // 最小绝对相似度（避免所有都不像）
    
    // 调试选项
    bool show_debug_windows = false;
    bool print_debug_info = false;

    // ROI 跟踪参数
    int roi_padding = 60;
    int roi_max_padding = 130;
    float roi_velocity_padding_gain = 0.8f;
    int roi_min_blob_area = 200;
    int roi_max_blob_area = 8000;
    int roi_hue_threshold = 12;   // 0-180
    int roi_sat_threshold = 60;   // 0-255
    int roi_val_threshold = 60;   // 0-255
    float roi_distance_score_weight = 1.2f;
    float roi_color_score_weight = 0.9f;
    float roi_area_score_weight = 0.4f;
    float roi_circularity_reject_threshold = 0.82f;
    float roi_center_reject_radius = 45.0f;
    float roi_max_step_pixels = 50.0f;  // 第一层稳定性改进：ROI最大单帧步长限制（像素）
    float roi_color_locking_threshold = 0.7f;  // 第二层稳定性改进：颜色锁定相似度阈值（0.0-1.0），防止误切到其他色块
    int roi_color_mismatch_tolerance = 3;      // 连续颜色不匹配多少帧后触发回退
    bool use_kalman = true;
    float kalman_dt = 1.0f / 30.0f;

    // 激光红点检测参数
    bool enable_laser_detection = true;
    int laser_hue_low_1 = 0;
    int laser_hue_high_1 = 12;
    int laser_hue_low_2 = 168;
    int laser_hue_high_2 = 179;
    int laser_min_saturation = 120;
    int laser_min_value = 120;
    int laser_min_blob_area = 3;
    int laser_max_blob_area = 2000;
    int laser_search_radius = 220;
    float laser_circularity_weight = 0.25f;

    // 距离估计参数
    // 使用像素面积与已知物理面积估计靶面到相机距离。
    float camera_fx_px = 381.625f;
    float camera_fy_px = 381.625f;
    bool enable_board_distance_estimation = true;
    // 经验标定系数：用于修正面积法带来的系统性偏差。
    // 例如实际 800mm 测得 500mm，可先设置为 1.6。
    float board_distance_calibration_scale = 1.0f;

    // 目标色块绕中心色块的物理半径（mm）
    float target_orbit_radius_mm = 200.0f;
};

// 色块信息结构体
struct ColorBlob {
    cv::Rect bounding_rect;      // 边界框
    cv::Point2f center;          // 中心点
    double area;                 // 面积
    double circularity;          // 圆形度
    cv::Scalar mean_color_bgr;   // 平均颜色 (BGR)
    cv::Scalar mean_color_hsv;   // 平均颜色 (HSV)
    bool is_dark;                // 是否为深色
};

// 跟踪结果结构体
struct TargetInfo {
    bool found = false;               // 是否找到目标
    cv::Point2f board_center;         // 标靶板中心
    cv::Point2f target_center;        // 目标色块中心
    float distance = 0.0f;            // 距离（像素）
    float angle = 0.0f;               // 角度（度）
    bool roi_active = false;          // 当前是否在使用ROI跟踪
    cv::Rect roi_rect{-1, -1, 0, 0};  // 当前ROI窗口
    bool laser_found = false;         // 是否找到激光红点
    cv::Point2f laser_center{-1.0f, -1.0f}; // 激光红点中心
    float laser_to_target_distance = 0.0f;  // 激光到目标的像素距离
    float board_distance_mm = -1.0f;  // 靶面到相机估计距离（mm）
    bool view_angle_valid = false;     // 是否有可用的角度偏移
    std::array<float, 1> view_delta_x{{0.0f}}; // X方向相对中心变化（yaw, rad）
    std::array<float, 1> view_delta_y{{0.0f}}; // Y方向相对中心变化（pitch, rad）
};

// 主跟踪器类
class TargetTracker {
public:
    using OrderedAngleArrays = std::pair<std::array<float, 1>, std::array<float, 1>>;

    /// 构造跟踪器并初始化默认配置与状态。
    TargetTracker();
    
    // 配置管理
    /// 覆盖当前跟踪参数配置。
    void set_config(const TrackerConfig& config);
    /// 返回当前生效的跟踪参数。
    TrackerConfig get_config() const;
    /// 一键启用/关闭调试输出与可视化窗口。
    void enable_debug(bool enabled);
    
    // 主处理函数
    /// 处理单帧图像并输出目标与激光检测结果。
    TargetInfo process_frame(const cv::Mat& frame);

    /// 基于目标距离与物理半径，计算相机相对中心应偏转的视角（rad）。
    /// 返回值为一对有序数组：first->x(yaw), second->y(pitch)。
    OrderedAngleArrays compute_view_angle_offsets(const cv::Point2f& board_center_px,
                                                  const cv::Point2f& target_center_px,
                                                  float board_distance_mm,
                                                  float target_orbit_radius_mm) const;

    // ROI 跟踪控制
    /// 清空ROI与历史跟踪状态，回到全图检测起点。
    void reset_roi_tracking();
    
    // 统计分析
    /// 重置帧计数与命中统计。
    void reset_statistics();
    /// 打印当前统计信息到控制台。
    void print_statistics() const;
    
private:
    // 核心处理函数
    /// 从图像中提取候选色块并输出调试掩码。
    std::vector<ColorBlob> extract_color_blobs(const cv::Mat& frame, cv::Mat& debug_mask);
    /// 在候选中选出最可能的中心色块。
    ColorBlob* find_center_blob(std::vector<ColorBlob>& blobs);
    /// 基于中心色块选择最终目标色块。
    ColorBlob* find_matching_target(const std::vector<ColorBlob>& blobs, const ColorBlob& center_blob);
    
    // 颜色匹配函数
    /// 计算两个颜色的综合相似度分数。
    double calculate_color_similarity(const cv::Scalar& color1, const cv::Scalar& color2, bool center_is_dark);
    
    // 辅助函数
    /// 根据轮廓面积和周长计算圆形度。
    double calculate_circularity(const std::vector<cv::Point>& contour);
    /// 判断候选色块是否满足中心周边几何约束。
    bool is_valid_surrounding_blob(const ColorBlob& blob, const ColorBlob& center);
    /// 将单个 BGR 颜色转换为 HSV。
    cv::Scalar bgr_to_hsv(const cv::Scalar& bgr);
    /// 将单个 BGR 颜色转换为 Lab。
    cv::Scalar bgr_to_lab(const cv::Scalar& bgr);
    /// 计算两个 BGR 颜色的欧氏距离。
    float color_distance_bgr(const cv::Scalar& c1, const cv::Scalar& c2);
    /// 依据亮度阈值判断颜色是否为暗色。
    bool is_dark_color(const cv::Scalar& bgr, int threshold);
    /// 对相似度列表做归一化，便于比较置信度。
    std::vector<double> normalize_similarities(const std::vector<double>& similarities);
    // 调试功能
    /// 在图像上绘制候选、中心与目标的调试信息。
    void draw_debug_info(cv::Mat& frame, 
                        const std::vector<ColorBlob>& blobs,
                        const ColorBlob* center,
                        const ColorBlob* target);
    
    // 成员变量
    TrackerConfig config_;
    cv::Size frame_size_;
    int frames_processed_;
    int successful_tracks_;
    bool has_previous_target_;
    cv::Point2f last_target_position_;
    cv::Point2f startup_target_position_;
    bool has_startup_target_position_ = false;
    cv::Point2f last_board_position_;
    bool has_previous_laser_ = false;
    cv::Point2f last_laser_position_{-1.0f, -1.0f};
    bool debug_enabled_;
    bool roi_tracking_active_ = false;
    cv::Scalar target_color_bgr_;
    cv::Scalar target_color_hsv_;
    // 第二层稳定性改进：当前ROI锚定的目标色块颜色信息
    cv::Scalar roi_target_color_bgr_locked_;
    cv::Scalar roi_target_color_hsv_locked_;
    bool roi_target_color_locked_ = false;       // 是否已锁定目标色块颜色
    int roi_color_mismatch_count_ = 0;           // 连续颜色不匹配计数
    cv::KalmanFilter kalman_;
    bool kalman_initialized_ = false;
    float estimated_board_distance_mm_ = -1.0f;
    struct ColorSimilarity {
        double bgr_sim;
        double hsv_sim;
        double lab_sim;
        double combined_sim;
    };
    /// 组合 BGR/HSV/Lab 多空间相似度并输出分项结果。
    ColorSimilarity calculate_multi_space_similarity(const cv::Scalar& color1, const cv::Scalar& color2, bool center_is_dark);
    /// 按 HSV 粗分类颜色类型用于动态加权。
    int classify_color_type(const cv::Scalar& hsv);
    /// 根据色块像素面积与已知物理尺寸估计靶面距离（mm）。
    float estimate_board_distance_mm_from_blobs(const std::vector<ColorBlob>& blobs) const;

    // ROI 跟踪实现
    /// 用当前目标初始化ROI与可选卡尔曼状态。
    bool init_roi_tracking(const cv::Mat& frame, const ColorBlob& target_blob);
    /// 在ROI内更新目标位置并回写结果。
    bool update_roi_tracking(const cv::Mat& frame, TargetInfo& result);
    /// 在给定ROI内执行目标色块检测。
    bool detect_target_in_roi(const cv::Mat& frame,
                              const cv::Rect& roi,
                              const cv::Point2f& expected_center_global,
                              cv::Point2f& out_center,
                              bool& rejected_by_circular_shape);
    /// 在目标附近检测激光红点位置。
    bool detect_laser_dot(const cv::Mat& frame,
                          const cv::Point2f& target_hint,
                          cv::Point2f& out_center,
                          cv::Mat* out_mask = nullptr);
};

#endif // TARGET_TRACKER_HPP