#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image_cv.hpp"
#include "main.h"

#include "TargetTracking/GeneticAlgorithm.hpp"
#include "TargetTracking/TargetTrackingPipeline.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace maix;

namespace {

constexpr float kPi = 3.14159265358979323846f;

struct Sample {
    float pitch_deg = 0.0f;
    float yaw_deg = 0.0f;
    float dt = 0.033f;
};

struct PidState {
    float integral = 0.0f;
    float prev_error = 0.0f;
    bool has_prev = false;
};

float clamp_value(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

float step_pid(float target,
               float current,
               float dt,
               const Genome &g,
               PidState &state,
               float integral_limit,
               float max_speed_deg)
{
    float error = target - current;
    state.integral += error * dt;
    state.integral = clamp_value(state.integral, -integral_limit, integral_limit);

    float derivative = 0.0f;
    if (state.has_prev && dt > 1e-6f) {
        derivative = (error - state.prev_error) / dt;
    }
    state.prev_error = error;
    state.has_prev = true;

    float speed = g.p * error + g.i * state.integral + g.d * derivative;
    speed = clamp_value(speed, -max_speed_deg, max_speed_deg);
    return current + speed * dt;
}

float evaluate_genome_on_samples(const Genome &g,
                                 const std::vector<Sample> &samples,
                                 float integral_limit,
                                 float max_speed_deg,
                                 float w_error,
                                 float w_smooth,
                                 float w_energy)
{
    if (samples.empty()) {
        return -1e9f;
    }

    float current_pitch = 0.0f;
    float current_yaw = 0.0f;
    float total_error = 0.0f;
    float total_smooth = 0.0f;
    float total_energy = 0.0f;

    float prev_speed_pitch = 0.0f;
    float prev_speed_yaw = 0.0f;

    PidState pitch_state;
    PidState yaw_state;

    for (const auto &s : samples) {
        float dt = clamp_value(s.dt, 0.001f, 0.05f);

        float prev_pitch = current_pitch;
        float prev_yaw = current_yaw;

        current_pitch = step_pid(s.pitch_deg, current_pitch, dt, g, pitch_state, integral_limit, max_speed_deg);
        current_yaw = step_pid(s.yaw_deg, current_yaw, dt, g, yaw_state, integral_limit, max_speed_deg);

        float speed_pitch = (current_pitch - prev_pitch) / dt;
        float speed_yaw = (current_yaw - prev_yaw) / dt;

        float err_pitch = std::abs(s.pitch_deg - current_pitch);
        float err_yaw = std::abs(s.yaw_deg - current_yaw);
        total_error += (err_pitch + err_yaw) * dt;

        total_smooth += (std::abs(speed_pitch - prev_speed_pitch) +
                         std::abs(speed_yaw - prev_speed_yaw)) * dt;

        total_energy += (std::abs(speed_pitch) + std::abs(speed_yaw)) * dt;

        prev_speed_pitch = speed_pitch;
        prev_speed_yaw = speed_yaw;
    }

    float cost = w_error * total_error +
                 w_smooth * total_smooth +
                 w_energy * total_energy;

    return -cost;
}

std::vector<Sample> collect_samples_from_pipeline(int max_frames,
                                                  bool show_preview,
                                                  TargetTrackingPipeline &pipeline,
                                                  camera::Camera &cam,
                                                  display::Display &disp,
                                                  float fx_hint,
                                                  float fy_hint)
{
    std::vector<Sample> samples;
    samples.reserve(max_frames);

    int collected = 0;
    int found_count = 0;

    uint64_t last_tick_ms = time::ticks_ms();

    while (!app::need_exit() && collected < max_frames) {
        image::Image *img = cam.read();
        if (!img) {
            time::sleep_ms(5);
            continue;
        }

        uint64_t now_tick_ms = time::ticks_ms();
        float dt = static_cast<float>(now_tick_ms - last_tick_ms) / 1000.0f;
        last_tick_ms = now_tick_ms;
        dt = clamp_value(dt, 0.001f, 0.05f);

    #ifdef MYGO_TARGETTRACKING_USE_MAIX
        PipelineOutput out = pipeline.process_frame(*img, dt);
    #else
        cv::Mat frame;
        maix::image::image2cv(*img, frame, true, true);
        PipelineOutput out = pipeline.process_frame(frame, dt);
    #endif
        if (out.target_found) {
            float fx = fx_hint > 0.0f ? fx_hint : static_cast<float>(img->width()) * 0.6f;
            float fy = fy_hint > 0.0f ? fy_hint : static_cast<float>(img->height()) * 0.6f;
            float cx = static_cast<float>(img->width()) * 0.5f;
            float cy = static_cast<float>(img->height()) * 0.5f;

            float dx = out.target_pos.x - cx;
            float dy = out.target_pos.y - cy;
            float pitch_deg = std::atan2(dy, fy) * 180.0f / kPi;
            float yaw_deg = -std::atan2(dx, fx) * 180.0f / kPi;

            Sample s;
            s.pitch_deg = pitch_deg;
            s.yaw_deg = yaw_deg;
            s.dt = dt;
            samples.push_back(s);
            found_count++;
        }

        if (show_preview) {
            img->draw_string(8, 8,
                             "collect=" + std::to_string(collected + 1) + "/" + std::to_string(max_frames),
                             image::COLOR_WHITE,
                             1.5f);
            img->draw_string(8, 30,
                             "found=" + std::to_string(found_count),
                             image::COLOR_GREEN,
                             1.5f);
            disp.show(*img, image::FIT_COVER);
        }

        delete img;
        collected++;
    }

    log::info("Collected samples: %d / %d", static_cast<int>(samples.size()), max_frames);
    return samples;
}

} // namespace

int _main(int argc, char *argv[])
{
    int collect_frames = 500;
    int population_size = 40;
    int generations = 120;
    bool show_preview = true;

    bool enable_serial = false;
    std::string serial_device = "/dev/ttyS0";
    int serial_baud = 115200;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            collect_frames = std::stoi(argv[++i]);
        } else if (arg == "--pop" && i + 1 < argc) {
            population_size = std::stoi(argv[++i]);
        } else if (arg == "--gen" && i + 1 < argc) {
            generations = std::stoi(argv[++i]);
        } else if (arg == "--no-preview") {
            show_preview = false;
        } else if (arg == "--serial" && i + 1 < argc) {
            serial_device = argv[++i];
            enable_serial = true;
        } else if (arg == "--baud" && i + 1 < argc) {
            serial_baud = std::stoi(argv[++i]);
        }
    }

    log::info("[mygo_ga_pipeline_irl] frames=%d pop=%d gen=%d preview=%s serial=%s baud=%d",
              collect_frames,
              population_size,
              generations,
              show_preview ? "on" : "off",
              enable_serial ? serial_device.c_str() : "off",
              serial_baud);

    camera::Camera cam(640, 480, image::Format::FMT_RGB888);
    display::Display disp;

    TargetTrackingPipeline pipeline;
    PipelineConfig cfg = pipeline.get_config();
    cfg.draw_overlay = false;
    cfg.print_debug = false;
    cfg.enable_serial = enable_serial;
    cfg.serial_device = serial_device;
    cfg.serial_baud = serial_baud;
    pipeline.set_config(cfg);

    if (enable_serial) {
        if (pipeline.open_serial()) {
            log::info("pipeline serial opened: %s @ %d", serial_device.c_str(), serial_baud);
        } else {
            log::warn("pipeline serial open failed, continue without output");
        }
    }

    cam.skip_frames(8);
    auto samples = collect_samples_from_pipeline(collect_frames,
                                                 show_preview,
                                                 pipeline,
                                                 cam,
                                                 disp,
                                                 -1.0f,
                                                 -1.0f);

    if (samples.size() < 30) {
        log::error("Not enough valid samples for GA training.");
        pipeline.close_serial();
        return -1;
    }

    Population population(population_size, 20260228);
    population.set_mutation(0.35f, 0.2f);
    population.set_elitism(2);

    const float integral_limit = 30.0f;
    const float max_speed_deg = 180.0f;
    const float w_error = 1.0f;
    const float w_smooth = 0.05f;
    const float w_energy = 0.01f;

    population.set_fitness_function([&](const Genome &g) {
        return evaluate_genome_on_samples(g,
                                          samples,
                                          integral_limit,
                                          max_speed_deg,
                                          w_error,
                                          w_smooth,
                                          w_energy);
    });

    Genome base;
    population.initialize_random(base);

    for (int gen = 0; gen < generations; ++gen) {
        population.evaluate_all();
        const Individual &best = population.best();
        log::info("Generation %d | Best Fitness: %.6f | P: %.6f, I: %.6f, D: %.6f",
                  population.generation(),
                  best.fitness,
                  best.genome.p,
                  best.genome.i,
                  best.genome.d);

        if (gen + 1 < generations) {
            population.evolve_next();
        }
    }

    population.evaluate_all();
    const Individual &best = population.best();
    log::info("Best PID => P: %.6f, I: %.6f, D: %.6f, Fitness: %.6f",
              best.genome.p,
              best.genome.i,
              best.genome.d,
              best.fitness);

    pipeline.close_serial();
    return 0;
}

int main(int argc, char *argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(_main, -1, argc, argv);
}
