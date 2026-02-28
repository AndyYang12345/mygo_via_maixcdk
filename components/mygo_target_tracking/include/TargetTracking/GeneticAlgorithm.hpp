#ifndef GENETIC_ALGORITHM_HPP
#define GENETIC_ALGORITHM_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

struct Genome {
    float p = 0.8f;
    float i = 0.0f;
    float d = 0.15f;

    float p_min = 0.0f;
    float p_max = 12.0f;
    float i_min = 0.0f;
    float i_max = 2.0f;
    float d_min = 0.0f;
    float d_max = 2.0f;

    void clamp() {
        p = std::max(p_min, std::min(p, p_max));
        i = std::max(i_min, std::min(i, i_max));
        d = std::max(d_min, std::min(d, d_max));
    }

    void mutate(std::mt19937& rng, float rate = 0.2f, float sigma = 0.1f) {
        std::normal_distribution<float> noise(0.0f, sigma);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        if (prob(rng) < rate) p += noise(rng);
        if (prob(rng) < rate) i += noise(rng);
        if (prob(rng) < rate) d += noise(rng);
        clamp();
    }

    static Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng) {
        std::uniform_real_distribution<float> mix(0.0f, 1.0f);
        float t = mix(rng);
        Genome out = a;
        out.p = a.p * t + b.p * (1.0f - t);
        out.i = a.i * t + b.i * (1.0f - t);
        out.d = a.d * t + b.d * (1.0f - t);
        out.p_min = a.p_min; out.p_max = a.p_max;
        out.i_min = a.i_min; out.i_max = a.i_max;
        out.d_min = a.d_min; out.d_max = a.d_max;
        out.clamp();
        return out;
    }
};

struct Individual {
    int id = -1;
    int generation = 0;
    Genome genome;
    float fitness = 0.0f;
    bool evaluated = false;
};

class Population {
public:
    using FitnessFunc = std::function<float(const Genome&)>;

    Population(int size, unsigned int seed = std::random_device{}())
        : _rng(seed), _size(size) {
        _individuals.reserve(size);
    }

    void set_fitness_function(FitnessFunc func) {
        _fitness = std::move(func);
    }

    struct TrackingFitnessConfig {
        struct Methods {
            int random = 0;
            int sine = 3;
            int circular = 2;
            int lissajous = 4;
        } methods;

        int targets_per_individual = 5;
        float dt = 0.01f;
        float max_time = 2.0f;
        float settle_threshold = 1.5f; // deg
        float settle_hold = 0.2f;       // sec
        float max_speed_deg = 180.0f;   // deg/s
        float integral_limit = 30.0f;
        float weight_time = 1.0f;
        float weight_error = 1.0f;
        float weight_smooth = 0.05f;
        float pitch_home = 60.0f;
        float yaw_home = 105.0f;
        float target_pitch_min = 45.0f;
        float target_pitch_max = 75.0f;
        float target_yaw_min = 75.0f;
        float target_yaw_max = 135.0f;
        float sine_amp_pitch = 15.0f;
        float sine_amp_yaw = 30.0f;
        float sine_frequency = 0.5f; // Hz
        float circular_radius_pitch = 12.0f;
        float circular_radius_yaw = 24.0f;
        float circular_frequency = 0.4f; // Hz
        float lissajous_a = 12.0f;
        float lissajous_b = 24.0f;
        float lissajous_wx = 1.0f; // rad/s
        float lissajous_wy = 1.7f; // rad/s
        float lissajous_phase = 0.0f;
        unsigned int seed = 12345;
    };

    void set_tracking_fitness(const TrackingFitnessConfig& cfg) {
        _tracking_config = cfg;
        _fitness = [this](const Genome& genome) {
            return evaluate_tracking_fitness(genome);
        };
    }

    void set_mutation(float rate, float sigma) {
        _mutation_rate = rate;
        _mutation_sigma = sigma;
    }

    void set_elitism(int elite_count) {
        _elite_count = std::max(0, elite_count);
    }

    void initialize_random(const Genome& base) {
        std::uniform_real_distribution<float> up(0.0f, 1.0f);
        _individuals.clear();
        for (int i = 0; i < _size; ++i) {
            Genome g = base;
            g.p = g.p_min + (g.p_max - g.p_min) * up(_rng);
            g.i = g.i_min + (g.i_max - g.i_min) * up(_rng);
            g.d = g.d_min + (g.d_max - g.d_min) * up(_rng);
            g.clamp();
            Individual ind;
            ind.id = i;
            ind.generation = _generation;
            ind.genome = g;
            ind.fitness = 0.0f;
            ind.evaluated = false;
            _individuals.push_back(ind);
        }
    }

    void evaluate_all() {
        if (!_fitness) {
            return;
        }
        for (auto& ind : _individuals) {
            if (!ind.evaluated) {
                ind.fitness = _fitness(ind.genome);
                ind.evaluated = true;
            }
        }
        sort_by_fitness();
    }

    const Individual& best() const {
        return _individuals.front();
    }

    const std::vector<Individual>& individuals() const {
        return _individuals;
    }

    int generation() const {
        return _generation;
    }

    void evolve_next() {
        if (_individuals.empty()) {
            return;
        }
        sort_by_fitness();

        std::vector<Individual> next;
        next.reserve(_size);

        int elite = std::min(_elite_count, static_cast<int>(_individuals.size()));
        for (int i = 0; i < elite; ++i) {
            Individual copy = _individuals[i];
            copy.generation = _generation + 1;
            copy.evaluated = false;
            next.push_back(copy);
        }

        while (static_cast<int>(next.size()) < _size) {
            const Individual& parent_a = tournament_select(3);
            const Individual& parent_b = tournament_select(3);
            Genome child = Genome::crossover(parent_a.genome, parent_b.genome, _rng);
            child.mutate(_rng, _mutation_rate, _mutation_sigma);
            Individual offspring;
            offspring.id = static_cast<int>(next.size());
            offspring.generation = _generation + 1;
            offspring.genome = child;
            offspring.fitness = 0.0f;
            offspring.evaluated = false;
            next.push_back(offspring);
        }

        _individuals = std::move(next);
        _generation += 1;
    }

private:
    std::mt19937 _rng;
    int _size = 0;
    int _generation = 0;
    float _mutation_rate = 0.2f;
    float _mutation_sigma = 0.1f;
    int _elite_count = 2;
    FitnessFunc _fitness;
    TrackingFitnessConfig _tracking_config;
    std::vector<Individual> _individuals;

    void sort_by_fitness() {
        std::sort(_individuals.begin(), _individuals.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });
    }

    const Individual& tournament_select(int k) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(_individuals.size()) - 1);
        int best_index = pick(_rng);
        for (int i = 1; i < k; ++i) {
            int idx = pick(_rng);
            if (_individuals[idx].fitness > _individuals[best_index].fitness) {
                best_index = idx;
            }
        }
        return _individuals[best_index];
    }

    struct PidState {
        float integral = 0.0f;
        float prev_error = 0.0f;
        bool has_prev = false;
    };

    using TargetFunc = std::function<std::pair<float, float>(float)>;

    float step_pid(float target, float current, float dt, const Genome& g, PidState& state) const {
        float error = target - current;
        state.integral += error * dt;
        state.integral = std::max(-_tracking_config.integral_limit,
                                  std::min(state.integral, _tracking_config.integral_limit));

        float derivative = 0.0f;
        if (state.has_prev && dt > 1e-6f) {
            derivative = (error - state.prev_error) / dt;
        }
        state.prev_error = error;
        state.has_prev = true;

        float rate = g.p * error + g.i * state.integral + g.d * derivative;
        rate = std::max(-_tracking_config.max_speed_deg, std::min(rate, _tracking_config.max_speed_deg));
        return current + rate * dt;
    }

    float simulate_trial(const Genome& genome, const TargetFunc& target_func, bool allow_settle) {
        float current_pitch = _tracking_config.pitch_home;
        float current_yaw = _tracking_config.yaw_home;
        float time_elapsed = 0.0f;
        float settle_timer = 0.0f;
        float error_integral = 0.0f;
        float smooth_penalty = 0.0f;
        float prev_rate_pitch = 0.0f;
        float prev_rate_yaw = 0.0f;

        PidState pitch_state;
        PidState yaw_state;

        while (time_elapsed < _tracking_config.max_time) {
            auto targets = target_func(time_elapsed);
            float target_pitch = targets.first;
            float target_yaw = targets.second;

            float prev_pitch = current_pitch;
            float prev_yaw = current_yaw;

            current_pitch = step_pid(target_pitch, current_pitch, _tracking_config.dt, genome, pitch_state);
            current_yaw = step_pid(target_yaw, current_yaw, _tracking_config.dt, genome, yaw_state);

            float rate_pitch = (current_pitch - prev_pitch) / _tracking_config.dt;
            float rate_yaw = (current_yaw - prev_yaw) / _tracking_config.dt;

            float dp = std::abs(target_pitch - current_pitch);
            float dy = std::abs(target_yaw - current_yaw);
            error_integral += (dp + dy) * _tracking_config.dt;

            smooth_penalty += (std::abs(rate_pitch - prev_rate_pitch) +
                               std::abs(rate_yaw - prev_rate_yaw)) * _tracking_config.dt;
            prev_rate_pitch = rate_pitch;
            prev_rate_yaw = rate_yaw;

            if (allow_settle) {
                if (dp < _tracking_config.settle_threshold && dy < _tracking_config.settle_threshold) {
                    settle_timer += _tracking_config.dt;
                    if (settle_timer >= _tracking_config.settle_hold) {
                        time_elapsed += _tracking_config.dt;
                        break;
                    }
                } else {
                    settle_timer = 0.0f;
                }
            }

            time_elapsed += _tracking_config.dt;
        }

        float cost = _tracking_config.weight_time * time_elapsed +
                     _tracking_config.weight_error * error_integral +
                     _tracking_config.weight_smooth * smooth_penalty;
        return cost;
    }

    float evaluate_tracking_fitness(const Genome& genome) {
        std::mt19937 eval_rng(_tracking_config.seed);
        std::uniform_real_distribution<float> pitch_dist(_tracking_config.target_pitch_min,
                                                         _tracking_config.target_pitch_max);
        std::uniform_real_distribution<float> yaw_dist(_tracking_config.target_yaw_min,
                                                       _tracking_config.target_yaw_max);

        float total_cost = 0.0f;
        const int random_trials = std::max(1,
            (_tracking_config.methods.random > 0) ?
            _tracking_config.methods.random : _tracking_config.targets_per_individual);

        for (int t = 0; t < random_trials; ++t) {
            float target_pitch = pitch_dist(eval_rng);
            float target_yaw = yaw_dist(eval_rng);
            total_cost += simulate_trial(
                genome,
                [target_pitch, target_yaw](float) {
                    return std::make_pair(target_pitch, target_yaw);
                },
                true);
        }

        for (int t = 0; t < _tracking_config.methods.sine; ++t) {
            float omega = 2.0f * static_cast<float>(M_PI) * _tracking_config.sine_frequency;
            total_cost += simulate_trial(
                genome,
                [this, omega](float time_s) {
                    float pitch = _tracking_config.pitch_home +
                                  _tracking_config.sine_amp_pitch * std::sin(omega * time_s);
                    float yaw = _tracking_config.yaw_home +
                                _tracking_config.sine_amp_yaw * std::sin(omega * time_s);
                    return std::make_pair(pitch, yaw);
                },
                false);
        }

        for (int t = 0; t < _tracking_config.methods.circular; ++t) {
            float omega = 2.0f * static_cast<float>(M_PI) * _tracking_config.circular_frequency;
            total_cost += simulate_trial(
                genome,
                [this, omega](float time_s) {
                    float pitch = _tracking_config.pitch_home +
                                  _tracking_config.circular_radius_pitch * std::cos(omega * time_s);
                    float yaw = _tracking_config.yaw_home +
                                _tracking_config.circular_radius_yaw * std::sin(omega * time_s);
                    return std::make_pair(pitch, yaw);
                },
                false);
        }

        for (int t = 0; t < _tracking_config.methods.lissajous; ++t) {
            total_cost += simulate_trial(
                genome,
                [this](float time_s) {
                    float pitch = _tracking_config.pitch_home + _tracking_config.lissajous_a *
                                  std::sin(_tracking_config.lissajous_wx * time_s +
                                           _tracking_config.lissajous_phase);
                    float yaw = _tracking_config.yaw_home + _tracking_config.lissajous_b *
                                std::sin(_tracking_config.lissajous_wy * time_s);
                    return std::make_pair(pitch, yaw);
                },
                false);
        }

        return -total_cost;
    }
};

#endif // GENETIC_ALGORITHM_HPP