// ==========================================================================
//  benchmark_pipeline - C++ twin of ../python/benchmark.py
//
//  Two-stage AI video pipeline benchmark:
//    Stage A: person detection (foot_track_net)
//    Stage B: PPE detection on cropped persons (gear_guard_net)
//
//  No CLI arguments - configuration lives in the constants below, so runs
//  are repeatable. The 5-stage pipeline mirrors the Python implementation:
//    reader+scale || model A || crop+prep B || model B || collector (main)
//
//  One run benchmarks TWO video-decode approaches back to back:
//    "software"  - CPU H.264 decode (openh264dec); the same decode class as
//                  Python's cv2.VideoCapture. Plain cv::VideoCapture is NOT
//                  used: on this platform its auto-picked HW decoder outputs
//                  compressed NV12 that videoconvert miscolors, breaking
//                  detection.
//    "gstreamer" - explicit GStreamer pipeline with the Qualcomm HW decoder
//                  (v4l2h264dec + qtivtransform)
//
//  Run from the experiment directory (run_cpp.sh does that):
//    ./build/benchmark_pipeline
// ==========================================================================

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <opencv2/opencv.hpp>

// LiteRT (ai-edge-litert): the TFLite C++ runtime now ships in the standalone
// LiteRT repo under the tflite/ prefix. The API is identical to the old
// tensorflow/lite headers (namespace tflite::, TfLiteExternalDelegate* C API) -
// only the include path changed.
#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"
#include "tflite/delegates/external/external_delegate.h"

namespace fs = std::filesystem;

// ===================================================================
// CONFIGURATION (mirrors the globals of benchmark.py)
// ===================================================================

static std::string VIDEO = "../input/1.mp4"; // overridable via $VIDEO (see main)
static const std::string MODEL_A = "../models/foot_track_net-tflite-w8a8/foot_track_net.tflite";
static const std::string MODEL_B = "../models/gear_guard_net-tflite-w8a8/gear_guard_net.tflite";
static const std::string DELEGATE = "hexagon"; // "hexagon" or "cpu"
static const std::string RESULTS_DIR = "results";
static const std::vector<std::string> DECODE_VARIANTS = {"software", "gstreamer"};

constexpr int NUM_THREADS = 4;
constexpr int HTP_PERF_MODE = 2;
constexpr int WARMUP_RUNS = 10;
constexpr int MAX_FRAMES = 0; // 0 = all
constexpr int QUEUE_SIZE = 8;

constexpr int TOPK_K = 1000;
constexpr int STRIDE = 4;
constexpr float PERSON_SCORE_THRESH = 0.76f;
constexpr float PERSON_IOU_THRESH = 0.5f;
constexpr float PPE_SCORE_THRESH = 0.40f;
constexpr float PPE_IOU_THRESH = 0.5f;

// ===================================================================
// SIGNAL HANDLING & TIMING
// ===================================================================

static std::atomic<bool> g_interrupted{false};
static void signal_handler(int) { g_interrupted.store(true, std::memory_order_relaxed); }

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

static double ms(TimePoint start, TimePoint end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ===================================================================
// DATA STRUCTURES
// ===================================================================

struct FrameMetrics
{
    int frame_id = 0;
    int num_persons_detected = 0;
    double decode_time_ms = 0.0;
    double prep_A_ms = 0.0;
    double infer_A_ms = 0.0;
    double post_A_ms = 0.0;
    double prep_B_ms = 0.0;
    double infer_B_ms = 0.0;
    double post_B_ms = 0.0;
    double total_frame_ms = 0.0;
    double cpu_usage_percent = 0.0;
    std::string per_core_cpu_percent;
    double ram_usage_mb = 0.0;
    double temp_C = 0.0;
};

struct TelemetrySnapshot
{
    double cpu_total = 0.0;
    std::vector<double> per_core;
    double ram_mb = 0.0;
    double temp_c = 0.0;
};

struct BBox
{
    int cls = 0;
    float score = 0.0f;
    float x, y, r, b;

    BBox(int c, float x_, float y_, float r_, float b_, float sc)
        : cls(c), score(sc),
          x(std::min(x_, r_)), y(std::min(y_, b_)),
          r(std::max(x_, r_)), b(std::max(y_, b_)) {}
};

// ===================================================================
// THREAD-SAFE BOUNDED QUEUE
// ===================================================================

template <typename T>
class BoundedQueue
{
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable not_full_, not_empty_;
    size_t max_size_;
    bool closed_ = false;

public:
    explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

    void push(T item)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [&]
                       { return q_.size() < max_size_ || closed_; });
        if (closed_)
            return;
        q_.push(std::move(item));
        not_empty_.notify_one();
    }

    bool pop(T &item)
    {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [&]
                        { return !q_.empty() || closed_; });
        if (q_.empty())
            return false;
        item = std::move(q_.front());
        q_.pop();
        not_full_.notify_one();
        return true;
    }

    void close()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        closed_ = true;
        not_full_.notify_all();
        not_empty_.notify_all();
    }
};

// ===================================================================
// TEMPERATURE & TELEMETRY (same sampling idea as common.py Telemetry)
// ===================================================================

static double read_soc_temperature()
{
    static const std::vector<std::string> PRIORITY_TYPES = {
        "cpu0-thermal", "cpuss0-thermal", "cpu-thermal"};
    const fs::path thermal_root("/sys/class/thermal");
    if (!fs::exists(thermal_root))
        return 0.0;

    std::map<std::string, fs::path> zone_map;
    try
    {
        for (const auto &entry : fs::directory_iterator(thermal_root))
        {
            auto type_path = entry.path() / "type";
            auto temp_path = entry.path() / "temp";
            if (!fs::exists(type_path) || !fs::exists(temp_path))
                continue;
            std::ifstream tf(type_path);
            std::string zone_type;
            if (std::getline(tf, zone_type))
            {
                while (!zone_type.empty() && std::isspace(zone_type.back()))
                    zone_type.pop_back();
                zone_map[zone_type] = temp_path;
            }
        }
    }
    catch (...)
    {
        return 0.0;
    }

    for (const auto &pref : PRIORITY_TYPES)
    {
        auto it = zone_map.find(pref);
        if (it == zone_map.end())
            continue;
        std::ifstream f(it->second);
        double raw = 0;
        if (f >> raw)
            return raw > 1000 ? raw / 1000.0 : raw;
    }
    return 0.0;
}

class TelemetryCollector
{
    std::thread thread_;
    std::atomic<bool> stop_{false};
    mutable std::mutex mtx_;
    TelemetrySnapshot latest_;
    double interval_sec_;

    struct CpuJiffies
    {
        long user = 0, nice = 0, system = 0, idle = 0;
        long iowait = 0, irq = 0, softirq = 0, steal = 0;
        long total() const { return user + nice + system + idle + iowait + irq + softirq + steal; }
        long busy() const { return total() - idle - iowait; }
    };
    CpuJiffies prev_total_;
    std::vector<CpuJiffies> prev_per_core_;
    bool first_read_ = true;

    static CpuJiffies parse_cpu_line(const std::string &line)
    {
        CpuJiffies j{};
        std::istringstream iss(line);
        std::string label;
        iss >> label >> j.user >> j.nice >> j.system >> j.idle >> j.iowait >> j.irq >> j.softirq >> j.steal;
        return j;
    }

    void sample(TelemetrySnapshot &snap)
    {
        std::ifstream f("/proc/stat");
        if (!f)
            return;

        std::string line;
        CpuJiffies total_now;
        std::vector<CpuJiffies> cores_now;
        while (std::getline(f, line))
        {
            if (line.compare(0, 4, "cpu ") == 0)
            {
                total_now = parse_cpu_line(line);
            }
            else if (line.compare(0, 3, "cpu") == 0 && std::isdigit(line[3]))
            {
                cores_now.push_back(parse_cpu_line(line));
            }
            else
            {
                break;
            }
        }

        if (first_read_)
        {
            first_read_ = false;
            prev_total_ = total_now;
            prev_per_core_ = cores_now;
            snap.per_core.assign(cores_now.size(), 0.0);
        }
        else
        {
            long dt = total_now.total() - prev_total_.total();
            long db = total_now.busy() - prev_total_.busy();
            snap.cpu_total = dt > 0 ? (100.0 * db / dt) : 0.0;
            snap.per_core.resize(cores_now.size());
            for (size_t i = 0; i < cores_now.size() && i < prev_per_core_.size(); i++)
            {
                long cdt = cores_now[i].total() - prev_per_core_[i].total();
                long cdb = cores_now[i].busy() - prev_per_core_[i].busy();
                snap.per_core[i] = cdt > 0 ? (100.0 * cdb / cdt) : 0.0;
            }
            prev_total_ = total_now;
            prev_per_core_ = cores_now;
        }

        std::ifstream statm("/proc/self/statm");
        long total_pages = 0, resident_pages = 0;
        statm >> total_pages >> resident_pages;
        snap.ram_mb = static_cast<double>(resident_pages) * sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
        snap.temp_c = read_soc_temperature();
    }

    void run()
    {
        while (!stop_.load(std::memory_order_relaxed))
        {
            TelemetrySnapshot snap;
            sample(snap);
            {
                std::lock_guard<std::mutex> lock(mtx_);
                latest_ = snap;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(interval_sec_ * 1000)));
        }
    }

public:
    explicit TelemetryCollector(double interval_sec = 0.2) : interval_sec_(interval_sec) {}
    void start() { thread_ = std::thread(&TelemetryCollector::run, this); }
    void stop()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable())
            thread_.join();
    }
    TelemetrySnapshot get_latest() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return latest_;
    }
};

// ===================================================================
// TFLITE INTERPRETER UTILITIES
// ===================================================================

struct InterpreterBundle
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate *ext_delegate = nullptr;

    ~InterpreterBundle()
    {
        interpreter.reset();
        if (ext_delegate)
        {
            TfLiteExternalDelegateDelete(ext_delegate);
            ext_delegate = nullptr;
        }
    }

    InterpreterBundle() = default;
    InterpreterBundle(InterpreterBundle &&) = default;
    InterpreterBundle &operator=(InterpreterBundle &&) = default;
    InterpreterBundle(const InterpreterBundle &) = delete;
    InterpreterBundle &operator=(const InterpreterBundle &) = delete;
};

static InterpreterBundle build_interpreter(const std::string &model_path)
{
    InterpreterBundle bundle;
    bundle.model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!bundle.model)
    {
        std::cerr << "ERROR: Failed to load model: " << model_path << "\n";
        std::exit(1);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*bundle.model, resolver);
    builder.SetNumThreads(NUM_THREADS);
    builder(&bundle.interpreter);
    if (!bundle.interpreter)
    {
        std::cerr << "ERROR: Failed to build interpreter for: " << model_path << "\n";
        std::exit(1);
    }

    if (DELEGATE == "hexagon")
    {
        TfLiteExternalDelegateOptions opts =
            TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");
        std::string perf_str = std::to_string(HTP_PERF_MODE);
        opts.insert(&opts, "backend_type", "htp");
        opts.insert(&opts, "htp_performance_mode", perf_str.c_str());

        bundle.ext_delegate = TfLiteExternalDelegateCreate(&opts);
        if (bundle.ext_delegate &&
            bundle.interpreter->ModifyGraphWithDelegate(bundle.ext_delegate) == kTfLiteOk)
        {
            std::cout << "  [delegate] Hexagon HTP loaded for "
                      << fs::path(model_path).filename().string() << "\n";
        }
        else
        {
            std::cerr << "  [delegate] WARN: Hexagon delegate unavailable, falling back to CPU\n";
            if (bundle.ext_delegate)
            {
                TfLiteExternalDelegateDelete(bundle.ext_delegate);
                bundle.ext_delegate = nullptr;
            }
        }
    }

    bundle.interpreter->AllocateTensors();
    return bundle;
}

static int find_output_by_name(tflite::Interpreter *interp, const std::string &name)
{
    for (int idx : interp->outputs())
    {
        const char *tname = interp->tensor(idx)->name;
        if (tname && name == tname)
            return idx;
    }
    std::cerr << "ERROR: Output tensor '" << name << "' not found.\n";
    std::exit(1);
}

static void dequantize_tensor_nhwc(tflite::Interpreter *interp, const std::string &name,
                                   std::vector<float> &out,
                                   int &out_h, int &out_w, int &out_c)
{
    int idx = find_output_by_name(interp, name);
    TfLiteTensor *t = interp->tensor(idx);
    int N = t->dims->data[0], H = t->dims->data[1];
    int W = t->dims->data[2], C = t->dims->data[3];
    out_h = H;
    out_w = W;
    out_c = C;

    int total = N * H * W * C;
    out.resize(total);
    float scale = t->params.scale;
    int zp = t->params.zero_point;

    if (t->type == kTfLiteUInt8)
    {
        const uint8_t *raw = interp->typed_tensor<uint8_t>(idx);
        for (int i = 0; i < total; i++)
            out[i] = (scale > 0.0f) ? (static_cast<float>(raw[i]) - zp) * scale : raw[i];
    }
    else if (t->type == kTfLiteInt8)
    {
        const int8_t *raw = interp->typed_tensor<int8_t>(idx);
        for (int i = 0; i < total; i++)
            out[i] = (scale > 0.0f) ? (static_cast<float>(raw[i]) - zp) * scale : raw[i];
    }
    else if (t->type == kTfLiteFloat32)
    {
        std::memcpy(out.data(), interp->typed_tensor<float>(idx), total * sizeof(float));
    }
    else
    {
        std::cerr << "ERROR: Unsupported tensor type for output '" << name << "'\n";
        std::exit(1);
    }
}

static std::vector<float> get_tensor_as_float(tflite::Interpreter *interp,
                                              const std::string &name, int &count)
{
    int idx = find_output_by_name(interp, name);
    TfLiteTensor *t = interp->tensor(idx);
    count = 1;
    for (int d = 0; d < t->dims->size; d++)
        count *= t->dims->data[d];

    float scale = t->params.scale;
    int zp = t->params.zero_point;
    std::vector<float> result(count);

    if (t->type == kTfLiteUInt8)
    {
        const uint8_t *raw = interp->typed_tensor<uint8_t>(idx);
        for (int i = 0; i < count; i++)
            result[i] = (scale > 0.0f) ? (static_cast<float>(raw[i]) - zp) * scale : raw[i];
    }
    else if (t->type == kTfLiteInt8)
    {
        const int8_t *raw = interp->typed_tensor<int8_t>(idx);
        for (int i = 0; i < count; i++)
            result[i] = (scale > 0.0f) ? (static_cast<float>(raw[i]) - zp) * scale : raw[i];
    }
    else if (t->type == kTfLiteFloat32)
    {
        std::memcpy(result.data(), interp->typed_tensor<float>(idx), count * sizeof(float));
    }
    else
    {
        std::cerr << "ERROR: Unsupported tensor type for '" << name << "'\n";
        std::exit(1);
    }
    return result;
}

// ===================================================================
// STAGE A POST-PROCESSING (heatmap-based detection, mirrors common.py)
// ===================================================================

static float get_iou(const BBox &a, const BBox &b)
{
    float xA = std::max(a.x, b.x);
    float yA = std::max(a.y, b.y);
    float xB = std::min(a.r, b.r);
    float yB = std::min(a.b, b.b);
    float inter = std::max(0.0f, xB - xA) * std::max(0.0f, yB - yA);
    float areaA = (a.r - a.x) * (a.b - a.y);
    float areaB = (b.r - b.x) * (b.b - b.y);
    return inter / (areaA + areaB - inter + 1e-5f);
}

static std::vector<BBox> nms(std::vector<BBox> &objs, float iou_thr)
{
    if (objs.size() <= 1)
        return objs;
    std::sort(objs.begin(), objs.end(),
              [](const BBox &a, const BBox &b)
              { return a.score > b.score; });
    std::vector<BBox> keep;
    for (const auto &obj : objs)
    {
        bool suppressed = false;
        for (const auto &k : keep)
        {
            if (get_iou(obj, k) > iou_thr)
            {
                suppressed = true;
                break;
            }
        }
        if (!suppressed)
            keep.push_back(obj);
    }
    return keep;
}

// heatmap [H,W,C] + tlrb [H,W,C*4] (NHWC, batch stripped) -> persons after NMS
static std::vector<BBox> detect_persons(const float *heatmap, int hm_h, int hm_w, int hm_c,
                                        const float *tlrb, int tlrb_c,
                                        std::vector<int> &scratch)
{
    constexpr int PERSON_CLS = 1;
    scratch.clear();
    for (int y = 0; y < hm_h; y++)
    {
        for (int x = 0; x < hm_w; x++)
        {
            if (heatmap[(y * hm_w + x) * hm_c + PERSON_CLS] > PERSON_SCORE_THRESH)
                scratch.push_back(y * hm_w + x);
        }
    }
    if (static_cast<int>(scratch.size()) > TOPK_K)
    {
        std::nth_element(scratch.begin(), scratch.begin() + TOPK_K, scratch.end(),
                         [&](int a, int b)
                         {
                             return heatmap[a * hm_c + PERSON_CLS] > heatmap[b * hm_c + PERSON_CLS];
                         });
        scratch.resize(TOPK_K);
    }

    std::vector<BBox> boxes;
    boxes.reserve(scratch.size());
    for (int pos : scratch)
    {
        int cy = pos / hm_w, cx = pos % hm_w;
        float score = heatmap[pos * hm_c + PERSON_CLS];
        int base = pos * tlrb_c + PERSON_CLS * 4;
        float left = tlrb[base + 0], top = tlrb[base + 1];
        float right = tlrb[base + 2], bottom = tlrb[base + 3];
        boxes.emplace_back(PERSON_CLS,
                           (cx - left) * STRIDE, (cy - top) * STRIDE,
                           (cx + right) * STRIDE, (cy + bottom) * STRIDE, score);
    }
    return nms(boxes, PERSON_IOU_THRESH);
}

// Raw stage B outputs copied out of the interpreter (so the next Invoke can
// start while a later pipeline stage post-processes them) - mirrors
// extract_ppe_outputs / detect_ppe_arrays in common.py.
struct PpeRaw
{
    std::vector<float> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
};

static std::vector<int> get_tensor_as_int(tflite::Interpreter *interp,
                                          const std::string &name)
{
    int idx = find_output_by_name(interp, name);
    TfLiteTensor *t = interp->tensor(idx);
    int total = 1;
    for (int d = 0; d < t->dims->size; d++)
        total *= t->dims->data[d];

    std::vector<int> result(total);
    if (t->type == kTfLiteUInt8)
    {
        const uint8_t *raw = interp->typed_tensor<uint8_t>(idx);
        for (int i = 0; i < total; i++)
            result[i] = raw[i];
    }
    else if (t->type == kTfLiteInt8)
    {
        const int8_t *raw = interp->typed_tensor<int8_t>(idx);
        for (int i = 0; i < total; i++)
            result[i] = raw[i];
    }
    else if (t->type == kTfLiteInt32)
    {
        const int32_t *raw = interp->typed_tensor<int32_t>(idx);
        for (int i = 0; i < total; i++)
            result[i] = raw[i];
    }
    else if (t->type == kTfLiteFloat32)
    {
        const float *raw = interp->typed_tensor<float>(idx);
        for (int i = 0; i < total; i++)
            result[i] = static_cast<int>(raw[i]);
    }
    return result;
}

static PpeRaw extract_ppe_raw(tflite::Interpreter *interp)
{
    PpeRaw raw;
    int n_boxes = 0, n_scores = 0;
    raw.boxes = get_tensor_as_float(interp, "boxes", n_boxes);
    raw.scores = get_tensor_as_float(interp, "scores", n_scores);
    raw.classes = get_tensor_as_int(interp, "class_idx");
    return raw;
}

// threshold + NMS on extracted outputs -> kept PPE detections (boxes in the
// model-input/padded-crop space; class 0=helmet, 1=vest).
static std::vector<BBox> ppe_nms(const PpeRaw &raw)
{
    std::vector<BBox> dets;
    int n = static_cast<int>(raw.scores.size());
    for (int i = 0; i < n; i++)
    {
        if (raw.scores[i] <= PPE_SCORE_THRESH)
            continue;
        dets.emplace_back(raw.classes[i], raw.boxes[i * 4 + 0], raw.boxes[i * 4 + 1],
                          raw.boxes[i * 4 + 2], raw.boxes[i * 4 + 3], raw.scores[i]);
    }
    if (dets.empty())
        return {};
    return nms(dets, PPE_IOU_THRESH);
}

static int ppe_nms_count(const PpeRaw &raw)
{
    return static_cast<int>(ppe_nms(raw).size());
}

// ===================================================================
// PREPROCESSING
// ===================================================================

struct PadResult
{
    float scale;
    int pad_w, pad_h;
};

static PadResult resize_and_pad(const cv::Mat &image, int target_h, int target_w,
                                cv::Mat &canvas, cv::Mat &scratch)
{
    int h = image.rows, w = image.cols;
    float scale = std::min(static_cast<float>(target_w) / w,
                           static_cast<float>(target_h) / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);

    cv::resize(image, scratch, cv::Size(new_w, new_h));
    canvas.setTo(0);
    int pad_w = (target_w - new_w) / 2;
    int pad_h = (target_h - new_h) / 2;
    scratch.copyTo(canvas(cv::Rect(pad_w, pad_h, new_w, new_h)));
    return {scale, pad_w, pad_h};
}

// Annotated-render helpers (SAVE_VIDEO=1). Colours match common.py (BGR):
// person (255,200,0), helmet (0,165,255), vest (0,255,0).
static void draw_box(cv::Mat &img, int x1, int y1, int x2, int y2,
                     const cv::Scalar &color, const std::string &label)
{
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2, cv::LINE_AA);
    cv::putText(img, label, cv::Point(x1 + 4, std::max(14, y1 - 6)),
                cv::FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv::LINE_AA);
}

// ===================================================================
// PIPELINE ITEM (one frame travelling through the 5 stages)
// ===================================================================

// Per-person crop geometry, kept only when rendering, so stage 5 can map PPE
// boxes back to full-frame coords (inverse of resize_and_pad) and draw them.
struct PersonBox
{
    int x1, y1, x2, y2; // person rect in full-frame pixels
    int pad_w, pad_h;   // letterbox padding used for its model-B crop
    float scale;        // resize_and_pad scale for its crop
};

struct PipelineItem
{
    bool valid = false;
    int frame_id = 0;
    double decode_ms = 0.0, prep_a_ms = 0.0, infer_a_ms = 0.0, post_a_ms = 0.0;
    double prep_b_ms = 0.0, infer_b_ms = 0.0, post_b_ms = 0.0;
    cv::Mat frame_bgr;
    cv::Mat small_rgb;
    std::vector<float> heatmap, bbox;
    int hm_h = 0, hm_w = 0, hm_c = 0, bb_c = 0;
    int num_persons = 0;
    std::vector<cv::Mat> canvases;
    std::vector<PpeRaw> ppe_raw;
    std::vector<PersonBox> persons_full; // parallel to canvases/ppe_raw (render only)
};

// ===================================================================
// WARMUP, STATS, CSV, SUMMARY
// ===================================================================

static void warmup(tflite::Interpreter *interp, int n_runs)
{
    int input_idx = interp->inputs()[0];
    std::memset(interp->tensor(input_idx)->data.raw, 0, interp->tensor(input_idx)->bytes);
    for (int i = 0; i < n_runs; i++)
        interp->Invoke();
}

struct Stats
{
    double mean, stddev, min_val, p50, p90, p99, max_val;
};

static Stats compute_stats(std::vector<double> vals)
{
    if (vals.empty())
        return {};
    std::sort(vals.begin(), vals.end());
    int n = static_cast<int>(vals.size());
    double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / n;
    double sq_sum = 0.0;
    for (double v : vals)
        sq_sum += (v - mean) * (v - mean);
    auto percentile = [&](double p)
    {
        double idx = p / 100.0 * (n - 1);
        int lo = static_cast<int>(idx);
        int hi = std::min(lo + 1, n - 1);
        double frac = idx - lo;
        return vals[lo] * (1.0 - frac) + vals[hi] * frac;
    };
    return {mean, std::sqrt(sq_sum / n), vals[0],
            percentile(50), percentile(90), percentile(99), vals[n - 1]};
}

static void export_csv(const std::string &path, const std::vector<FrameMetrics> &metrics)
{
    std::ofstream f(path);
    if (!f)
    {
        std::cerr << "ERROR: Cannot write CSV to: " << path << "\n";
        return;
    }
    f << "frame_id,num_persons_detected,"
      << "decode_time_ms,prep_A_ms,infer_A_ms,post_A_ms,"
      << "prep_B_ms,infer_B_ms,post_B_ms,total_frame_ms,"
      << "cpu_usage_percent,per_core_cpu_percent,ram_usage_mb,temp_C\n";
    f << std::setprecision(15);
    for (const auto &m : metrics)
    {
        f << m.frame_id << "," << m.num_persons_detected << ","
          << m.decode_time_ms << "," << m.prep_A_ms << "," << m.infer_A_ms << ","
          << m.post_A_ms << "," << m.prep_B_ms << "," << m.infer_B_ms << ","
          << m.post_B_ms << "," << m.total_frame_ms << "," << m.cpu_usage_percent << ","
          << m.per_core_cpu_percent << "," << m.ram_usage_mb << "," << m.temp_C << "\n";
    }
}

static void write_summary(const std::string &path, const std::string &decode,
                          int frames, double duration_s)
{
    std::ofstream f(path);
    if (!f)
    {
        std::cerr << "ERROR: Cannot write summary to: " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(2);
    f << "{\n"
      << "  \"implementation\": \"cpp\",\n"
      << "  \"mode\": \"pipelined\",\n"
      << "  \"decode\": \"" << decode << "\",\n"
      << "  \"delegate\": \"" << DELEGATE << "\",\n"
      << "  \"video\": \"" << VIDEO << "\",\n"
      << "  \"frames\": " << frames << ",\n"
      << "  \"duration_s\": " << duration_s << ",\n"
      << "  \"fps\": " << (duration_s > 0 ? frames / duration_s : 0.0) << "\n"
      << "}\n";
}

static void print_summary(const std::vector<FrameMetrics> &metrics, double wall_s)
{
    auto arr = [&](auto getter)
    {
        std::vector<double> v;
        v.reserve(metrics.size());
        for (const auto &m : metrics)
            v.push_back(getter(m));
        return v;
    };

    std::cout << "\n"
              << std::string(70, '=') << "\n";
    std::cout << "  BENCHMARK RESULTS (C++ 5-stage pipeline)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Frames processed : " << metrics.size() << "\n";
    std::cout << "  Wall-clock time  : " << wall_s << " s\n";
    std::cout << "  Average FPS      : " << (wall_s > 0 ? metrics.size() / wall_s : 0.0) << "\n\n";

    struct StageInfo
    {
        const char *label;
        double (*getter)(const FrameMetrics &);
    };
    const StageInfo stages[] = {
        {"Decode           ", [](const FrameMetrics &m)
         { return m.decode_time_ms; }},
        {"Preprocess A     ", [](const FrameMetrics &m)
         { return m.prep_A_ms; }},
        {"Inference A      ", [](const FrameMetrics &m)
         { return m.infer_A_ms; }},
        {"Postprocess A    ", [](const FrameMetrics &m)
         { return m.post_A_ms; }},
        {"Preprocess B     ", [](const FrameMetrics &m)
         { return m.prep_B_ms; }},
        {"Inference B      ", [](const FrameMetrics &m)
         { return m.infer_B_ms; }},
        {"Postprocess B    ", [](const FrameMetrics &m)
         { return m.post_B_ms; }},
        {"Compute total    ", [](const FrameMetrics &m)
         { return m.total_frame_ms; }},
    };

    printf("  %-18s  %8s  %8s  %8s  %8s  %8s  %8s  %8s\n",
           "Stage", "Mean", "Std", "Min", "P50", "P90", "P99", "Max");
    printf("  %s\n", std::string(90, '-').c_str());
    for (const auto &si : stages)
    {
        Stats s = compute_stats(arr(si.getter));
        printf("  %-18s  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f  %8.2f\n",
               si.label, s.mean, s.stddev, s.min_val, s.p50, s.p90, s.p99, s.max_val);
    }
    std::cout << std::string(70, '=') << "\n";
}

// ===================================================================
// MAIN
// ===================================================================

int main()
{
    signal(SIGINT, signal_handler);

    // Environment overrides (keep the "no CLI args" convention; the host
    // orchestrator sets these over ssh):
    //   LOOP_RUNS=N        replay the input N times back-to-back (default 1) -
    //                      a long-enough run for a steady power trace + thermal
    //                      soak, instead of the ~3-6 s single pass.
    //   DECODE_VARIANT=x   run only decode variant "software" or "gstreamer"
    //                      (default: both, back to back) so each can get its
    //                      own power recording.
    int loop_runs = 1;
    if (const char *lr = std::getenv("LOOP_RUNS"); lr && *lr)
    {
        loop_runs = std::max(1, std::atoi(lr));
    }
    //   RUN_SECONDS=S      OVERRIDES LOOP_RUNS: replay the clip in-process until
    //                      S seconds of wall-clock elapse. The long-term session
    //                      sets it so ONE invocation holds the load for the whole
    //                      window (models stay loaded - no reload blip), giving a
    //                      clean continuous trace instead of the re-invocation
    //                      sawtooth.
    double run_seconds = 0.0;
    if (const char *rs = std::getenv("RUN_SECONDS"); rs && *rs)
    {
        run_seconds = std::atof(rs);
    }
    //   VIDEO=path         input clip (default input/1.mp4). The long-term
    //                      session points this at one long looped clip in
    //                      /dev/shm so all three variants share the same input.
    if (const char *vv = std::getenv("VIDEO"); vv && *vv)
    {
        VIDEO = vv;
    }
    //   SAVE_VIDEO=1       burn person/helmet/vest boxes into an annotated mp4
    //                      (results/cpp_<variant>_boxes.mp4) - for the short
    //                      visual check; off for the perf/thermal runs.
    bool render = false;
    if (const char *sv = std::getenv("SAVE_VIDEO"); sv && *sv && std::string(sv) != "0")
        render = true;
    std::vector<std::string> variants = DECODE_VARIANTS;
    if (const char *dv = std::getenv("DECODE_VARIANT"); dv && *dv)
    {
        variants = {dv};
    }

    std::cout << std::string(70, '=') << "\n";
    std::cout << "  TWO-STAGE PIPELINE BENCHMARK (C++ threaded twin of benchmark.py)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Delegate : " << DELEGATE << " | Threads: " << NUM_THREADS
              << " | HTP perf mode: " << HTP_PERF_MODE << "\n";
    if (run_seconds > 0.0)
        std::cout << "  Video    : " << VIDEO << "  (run " << run_seconds << "s)\n";
    else
        std::cout << "  Video    : " << VIDEO << "  (loop x" << loop_runs << ")\n";

    // --- Interpreters + warmup.
    auto bundle_a = build_interpreter(MODEL_A);
    auto bundle_b = build_interpreter(MODEL_B);
    auto *interp_a = bundle_a.interpreter.get();
    auto *interp_b = bundle_b.interpreter.get();

    int h_a = interp_a->tensor(interp_a->inputs()[0])->dims->data[1];
    int w_a = interp_a->tensor(interp_a->inputs()[0])->dims->data[2];
    int h_b = interp_b->tensor(interp_b->inputs()[0])->dims->data[1];
    int w_b = interp_b->tensor(interp_b->inputs()[0])->dims->data[2];

    std::cout << "  Warmup   : " << WARMUP_RUNS << " runs per model ...\n";
    warmup(interp_a, WARMUP_RUNS);
    warmup(interp_b, WARMUP_RUNS);

    fs::create_directories(RESULTS_DIR);

    TelemetryCollector telemetry(0.2);
    telemetry.start();

    // --- One decode variant: open capture (AFTER warmup - a live appsink
    //     pipeline starts decoding as soon as it opens), run the 5-stage
    //     pipeline, export per-variant CSV + summary ---
    auto open_capture = [&](const std::string &variant) -> cv::VideoCapture
    {
        cv::VideoCapture cap;
        // drop=false + max-buffers gives backpressure: every frame is
        // delivered, none discarded when the consumer is slower
        const std::string sink = " ! appsink max-buffers=4 drop=false sync=false";
        auto try_gst = [&](const std::string &pipe) -> bool
        {
            cap.open(pipe, cv::CAP_GSTREAMER);
            if (cap.isOpened())
                return true;
            cap.release();
            return false;
        };
        if (variant == "software")
        {
            if (!try_gst("filesrc location=" + VIDEO + " ! qtdemux ! h264parse ! "
                                                       "openh264dec ! videoconvert ! video/x-raw,format=BGR" +
                         sink) &&
                !try_gst("filesrc location=" + VIDEO + " ! qtdemux ! h264parse ! "
                                                       "avdec_h264 ! videoconvert ! video/x-raw,format=BGR" +
                         sink))
            {
                cap.open(VIDEO);
            }
            return cap;
        }
        // HW decode path (v4l2h264dec on the Qualcomm codec). The colour
        // conversion MUST go through BGRx, not straight to BGR: qtivtransform is
        // a GPU converter and only emits a cleanly-packed buffer in a 4-byte
        // aligned format (BGRx). Asking it for 24-bit BGR directly yields a
        // buffer OpenCV's appsink mis-strides - the frame arrives replicated
        // horizontally + interlaced, so detection sees noise (0 persons). So:
        // qtivtransform -> BGRx (GPU, aligned) -> videoconvert -> BGR (cheap CPU
        // repack). Decode itself stays on the HW v4l2h264dec.
        try_gst("filesrc location=" + VIDEO + " ! qtdemux ! h264parse ! v4l2h264dec ! "
                                              "qtivtransform ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR" +
                sink) ||
            try_gst("filesrc location=" + VIDEO + " ! decodebin ! qtivtransform ! "
                                                  "video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR" +
                    sink);
        return cap;
    };

    bool any_variant_ok = false;
    for (const std::string &variant : variants)
    {
        std::cout << "\n"
                  << std::string(70, '-') << "\n";
        std::cout << "  Decode variant: " << variant << "  (loop x" << loop_runs << ")\n";
        std::cout << std::string(70, '-') << "\n";

        // Probe-open once to validate the decoder and read the frame size; the
        // reader thread reopens a fresh capture for each loop iteration (an
        // appsink capture can't be rewound reliably).
        cv::VideoCapture probe = open_capture(variant);
        if (!probe.isOpened())
        {
            std::cerr << "  WARN: cannot open " << VIDEO << " with the '" << variant
                      << "' decoder - skipping this variant.\n";
            continue;
        }

        int orig_w = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_WIDTH));
        int orig_h = static_cast<int>(probe.get(cv::CAP_PROP_FRAME_HEIGHT));
        probe.release();

        std::vector<FrameMetrics> all_metrics;
        float scale_x = static_cast<float>(orig_w) / w_a;
        float scale_y = static_cast<float>(orig_h) / h_a;

        BoundedQueue<PipelineItem> q1(QUEUE_SIZE), q2(QUEUE_SIZE),
            q3(QUEUE_SIZE), q4(QUEUE_SIZE);

        // Decoder/pipeline open time is accumulated by the reader and subtracted
        // from the wall clock below, so FPS is steady-state processing throughput,
        // not penalised by the (heavy, for HW decode: EGL/GPU) per-pass capture
        // init. Matches the amortised rate a long continuous run already reports.
        double reader_open_ms = 0.0;
        auto t_bench_start = Clock::now();

        // Stage 1: read frame + scale for model A. Replays the clip loop_runs
        // times, reopening a fresh capture each pass (appsink can't be rewound).
        std::thread t_reader([&]
                             {
        int fid = 0;
        cv::Mat small_bgr(h_a, w_a, CV_8UC3);
        // Time cap is checked PER FRAME (not just per file loop) so it stops on
        // the dot whether VIDEO is the short clip (re-looped) or the long
        // /dev/shm clip (read once, mid-file).
        auto time_up = [&] {
            return run_seconds > 0.0 && ms(t_bench_start, Clock::now()) >= run_seconds * 1000.0;
        };
        for (int loop = 0; !g_interrupted.load(std::memory_order_relaxed) && !time_up(); loop++) {
            auto t_open = Clock::now();
            cv::VideoCapture cap = open_capture(variant);
            reader_open_ms += ms(t_open, Clock::now());   // excluded from wall FPS
            if (!cap.isOpened()) break;
            while (!g_interrupted.load(std::memory_order_relaxed) && !time_up()) {
                if (MAX_FRAMES > 0 && fid >= MAX_FRAMES) break;
                PipelineItem item;
                auto t0 = Clock::now();
                if (!cap.read(item.frame_bgr)) break;
                item.decode_ms = ms(t0, Clock::now());
                auto t1 = Clock::now();
                cv::resize(item.frame_bgr, small_bgr, cv::Size(w_a, h_a), 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(small_bgr, item.small_rgb, cv::COLOR_BGR2RGB);
                item.prep_a_ms = ms(t1, Clock::now());
                item.valid = true;
                item.frame_id = fid++;
                q1.push(std::move(item));
            }
            cap.release();
            if (MAX_FRAMES > 0 && fid >= MAX_FRAMES) break;
            // Without a time budget, stop after loop_runs replays (short test).
            if (run_seconds <= 0.0 && loop + 1 >= loop_runs) break;
        }
        q1.push(PipelineItem{}); });

        // Stage 2: model A inference + output extraction
        std::thread t_model_a([&]
                              {
        PipelineItem item;
        while (q1.pop(item) && item.valid) {
            auto t0 = Clock::now();
            std::memcpy(interp_a->typed_input_tensor<uint8_t>(0), item.small_rgb.data,
                        item.small_rgb.total() * item.small_rgb.elemSize());
            item.prep_a_ms += ms(t0, Clock::now());
            auto t1 = Clock::now();
            interp_a->Invoke();
            auto t2 = Clock::now();
            item.infer_a_ms = ms(t1, t2);
            int bb_h, bb_w;
            dequantize_tensor_nhwc(interp_a, "heatmap", item.heatmap,
                                   item.hm_h, item.hm_w, item.hm_c);
            dequantize_tensor_nhwc(interp_a, "bbox", item.bbox, bb_h, bb_w, item.bb_c);
            item.post_a_ms = ms(t2, Clock::now());
            item.small_rgb.release();
            q2.push(std::move(item));
        }
        q2.push(PipelineItem{}); });

        // Stage 3: person detection + crop/letterbox for model B
        std::thread t_crop([&]
                           {
        std::vector<int> topk_scratch;
        topk_scratch.reserve(4096);
        cv::Mat crop_rgb, resize_scratch;
        PipelineItem item;
        while (q2.pop(item) && item.valid) {
            auto t0 = Clock::now();
            auto persons = detect_persons(item.heatmap.data(), item.hm_h, item.hm_w,
                                          item.hm_c, item.bbox.data(), item.bb_c,
                                          topk_scratch);
            item.post_a_ms += ms(t0, Clock::now());
            item.heatmap.clear();
            item.bbox.clear();

            for (const auto& p : persons) {
                int x1 = std::max(0, static_cast<int>(p.x * scale_x));
                int y1 = std::max(0, static_cast<int>(p.y * scale_y));
                int x2 = std::min(orig_w, static_cast<int>(p.r * scale_x));
                int y2 = std::min(orig_h, static_cast<int>(p.b * scale_y));
                if (x2 <= x1 || y2 <= y1) continue;

                auto t1 = Clock::now();
                cv::Mat crop_bgr = item.frame_bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                cv::cvtColor(crop_bgr, crop_rgb, cv::COLOR_BGR2RGB);
                cv::Mat canvas(h_b, w_b, CV_8UC3, cv::Scalar(0));
                PadResult pr = resize_and_pad(crop_rgb, h_b, w_b, canvas, resize_scratch);
                item.prep_b_ms += ms(t1, Clock::now());
                item.canvases.push_back(std::move(canvas));
                if (render)
                    item.persons_full.push_back({x1, y1, x2, y2, pr.pad_w, pr.pad_h, pr.scale});
            }
            item.num_persons = static_cast<int>(item.canvases.size());
            if (!render) item.frame_bgr.release();   // keep the frame for drawing when rendering
            q3.push(std::move(item));
        }
        q3.push(PipelineItem{}); });

        // Stage 4: model B inference + output extraction (per person, sequential -
        // a parallel interpreter pool is impossible here, see the interpreter note).
        std::thread t_model_b([&]
                              {
        PipelineItem item;
        while (q3.pop(item) && item.valid) {
            for (auto& canvas : item.canvases) {
                auto t0 = Clock::now();
                std::memcpy(interp_b->typed_input_tensor<uint8_t>(0), canvas.data,
                            canvas.total() * canvas.elemSize());
                item.prep_b_ms += ms(t0, Clock::now());
                auto t1 = Clock::now();
                interp_b->Invoke();
                auto t2 = Clock::now();
                item.infer_b_ms += ms(t1, t2);
                item.ppe_raw.push_back(extract_ppe_raw(interp_b));
                item.post_b_ms += ms(t2, Clock::now());
            }
            item.canvases.clear();
            q4.push(std::move(item));
        }
        q4.push(PipelineItem{}); });

        // Stage 5 (main thread): PPE postprocess + metrics collection
        cv::VideoWriter writer; // opened lazily on the first frame when rendering
        PipelineItem item;
        while (q4.pop(item) && item.valid)
        {
            for (const auto &raw : item.ppe_raw)
            {
                auto t0 = Clock::now();
                ppe_nms_count(raw);
                item.post_b_ms += ms(t0, Clock::now());
            }

            // Annotated render (SAVE_VIDEO=1): person + helmet/vest boxes -> mp4.
            // PPE boxes are in padded-crop space; map back to full frame (inverse of
            // resize_and_pad), mirroring common.ppe_to_global.
            if (render && !item.frame_bgr.empty())
            {
                for (size_t i = 0; i < item.persons_full.size() && i < item.ppe_raw.size(); i++)
                {
                    const PersonBox &pb = item.persons_full[i];
                    draw_box(item.frame_bgr, pb.x1, pb.y1, pb.x2, pb.y2,
                             cv::Scalar(255, 200, 0), "Person");
                    for (const BBox &d : ppe_nms(item.ppe_raw[i]))
                    {
                        int gx1 = std::max(0, pb.x1 + static_cast<int>((d.x - pb.pad_w) / pb.scale));
                        int gy1 = std::max(0, pb.y1 + static_cast<int>((d.y - pb.pad_h) / pb.scale));
                        int gx2 = std::min(orig_w, pb.x1 + static_cast<int>((d.r - pb.pad_w) / pb.scale));
                        int gy2 = std::min(orig_h, pb.y1 + static_cast<int>((d.b - pb.pad_h) / pb.scale));
                        bool helmet = (d.cls == 0);
                        draw_box(item.frame_bgr, gx1, gy1, gx2, gy2,
                                 helmet ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0),
                                 helmet ? "Helmet" : "Vest");
                    }
                }
                if (!writer.isOpened())
                    writer.open(RESULTS_DIR + "/cpp_" + variant + "_boxes.mp4",
                                cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25.0,
                                cv::Size(orig_w, orig_h));
                writer.write(item.frame_bgr);
            }

            auto snap = telemetry.get_latest();

            FrameMetrics m;
            m.frame_id = item.frame_id;
            m.num_persons_detected = item.num_persons;
            m.decode_time_ms = item.decode_ms;
            m.prep_A_ms = item.prep_a_ms;
            m.infer_A_ms = item.infer_a_ms;
            m.post_A_ms = item.post_a_ms;
            m.prep_B_ms = item.prep_b_ms;
            m.infer_B_ms = item.infer_b_ms;
            m.post_B_ms = item.post_b_ms;
            m.total_frame_ms = item.prep_a_ms + item.infer_a_ms + item.post_a_ms + item.prep_b_ms + item.infer_b_ms + item.post_b_ms;
            m.cpu_usage_percent = snap.cpu_total;
            {
                std::ostringstream oss;
                for (size_t i = 0; i < snap.per_core.size(); i++)
                {
                    if (i > 0)
                        oss << "|";
                    oss << std::fixed << std::setprecision(0) << snap.per_core[i];
                }
                m.per_core_cpu_percent = oss.str();
            }
            m.ram_usage_mb = snap.ram_mb;
            m.temp_C = snap.temp_c;
            all_metrics.push_back(std::move(m));
        }

        if (writer.isOpened())
            writer.release();

        double wall_raw_ms = ms(t_bench_start, Clock::now());

        t_reader.join();
        t_model_a.join();
        t_crop.join();
        t_model_b.join();

        // Subtract decoder-open time (reader is now joined, so reader_open_ms is
        // final) -> a short looped sprint isn't taxed by repeated capture init.
        double wall_s = std::max(0.0, wall_raw_ms - reader_open_ms) / 1000.0;

        if (all_metrics.empty())
        {
            std::cerr << "  WARN: no frames processed for variant '" << variant << "'.\n";
            continue;
        }
        any_variant_ok = true;

        std::string csv_path = RESULTS_DIR + "/cpp_" + variant + "_metrics.csv";
        std::string sum_path = RESULTS_DIR + "/summary_cpp_" + variant + ".json";
        export_csv(csv_path, all_metrics);
        write_summary(sum_path, variant, static_cast<int>(all_metrics.size()), wall_s);
        printf("\n  CSV saved to: %s  (%zu rows)\n", csv_path.c_str(), all_metrics.size());
        printf("  Summary saved to: %s\n", sum_path.c_str());
        print_summary(all_metrics, wall_s);
    } // variant loop

    telemetry.stop();
    return any_variant_ok ? 0 : 1;
}
