/*
 * Custom qtimlpostprocess module for gear_guard_net (PPE detection).
 *
 * Why this exists: gear_guard_net-ppe-detection-w8a8 (Qualcomm AI Hub, open
 * license) emits THREE already-decoded, pre-NMS tensors:
 *     boxes     [1, N, 4]   (x1, y1, x2, y2 in model-input pixels)
 *     scores    [1, N]      (max class score per anchor)
 *     class_idx [1, N]      (argmax class per anchor)
 * No stock qti module consumes that layout: yolov5/yolov8 want a single
 * [1, 4+C, N] tensor, yolo-nas wants per-class scores [1,N,C], ssd-mobilenet
 * wants a separate count tensor. This module does the only thing missing:
 * threshold + per-class NMS on the decoded outputs, mirroring the proven
 * postprocessing.
 *
 * Built as a standalone .so (dlopen'd by qtimlpostprocess via module=gearguard).
 * SPDX-License-Identifier: BSD-3-Clause-Clear
 */

#include "qti-ml-post-process.h"
#include "qti-labels-parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

static const float kDefaultThreshold        = 0.40f;
static const float kNMSIntersectionThreshold  = 0.5f;

// gear_guard_net-ppe-detection-w8a8: 3780 anchors at 192x320 input.
static const std::string kModuleCaps = R"(
{
  "type": "object-detection",
  "tensors": [
    {
      "format": ["FLOAT32"],
      "dimensions": [ [1, 3780, 4], [1, 3780], [1, 3780] ]
    }
  ]
}
)";

class Module : public IModule {
 public:
  Module(LogCallback cb) : logger_(cb), threshold_(kDefaultThreshold) {}
  ~Module() {}

  std::string Caps() override { return kModuleCaps; }

  bool Configure(const std::string& labels_file,
                 const std::string& json_settings) override {
    if (!labels_parser_.LoadFromFile(labels_file)) {
      LOG(logger_, kError, "gearguard: failed to parse labels");
      return false;
    }
    if (!json_settings.empty()) {
      auto root = JsonValue::Parse(json_settings);
      if (!root || root->GetType() != JsonType::Object) return false;
      threshold_ = root->GetNumber("confidence") / 100.0;
      LOG(logger_, kLog, "gearguard: threshold %f", threshold_);
    }
    return true;
  }

  bool Process(const Tensors& tensors, Dictionary& mlparams,
               std::any& output) override {
    if (output.type() != typeid(ObjectDetections)) {
      LOG(logger_, kError, "gearguard: unexpected predictions type");
      return false;
    }
    ObjectDetections& detections = std::any_cast<ObjectDetections&>(output);
    Region& region = std::any_cast<Region&>(mlparams["input-tensor-region"]);

    // Identify the three tensors by shape/name (order is not guaranteed):
    //   boxes     -> 3 dims, last == 4
    //   class_idx -> 2 dims, name contains "class"
    //   scores    -> the remaining 2-dim tensor
    const float *boxes = nullptr, *scores = nullptr, *classes = nullptr;
    uint32_t n_boxes = 0;
    for (const auto& t : tensors) {
      if (t.dimensions.size() == 3 && t.dimensions.back() == 4) {
        boxes = reinterpret_cast<const float*>(t.data);
        n_boxes = t.dimensions[1];
      } else if (t.dimensions.size() == 2) {
        if (t.name.find("class") != std::string::npos)
          classes = reinterpret_cast<const float*>(t.data);
        else
          scores = reinterpret_cast<const float*>(t.data);
      }
    }
    if (!boxes || !scores || !classes) {
      LOG(logger_, kError, "gearguard: missing boxes/scores/class_idx tensors");
      return false;
    }

    for (uint32_t idx = 0; idx < n_boxes; idx++) {
      if (scores[idx] < threshold_)
        continue;

      ObjectDetection entry;
      entry.left   = boxes[idx * 4 + 0];
      entry.top    = boxes[idx * 4 + 1];
      entry.right  = boxes[idx * 4 + 2];
      entry.bottom = boxes[idx * 4 + 3];

      // Map model-input pixels to region-relative [0..1] (qti convention).
      TransformDimensions(entry, region);

      // Drop spurious detections that fall outside the person region (or are
      // degenerate) rather than clamping them. Clamping pins an out-of-frame
      // box to an edge -> a stray filled box in the frame corner. Matches the
      // stock ssd-mobilenet module's reject-if->1.0 behaviour.
      if (entry.top  > 1.0f || entry.left   > 1.0f ||
          entry.bottom > 1.0f || entry.right  > 1.0f ||
          entry.left < 0.0f || entry.top    < 0.0f ||
          entry.right <= entry.left || entry.bottom <= entry.top)
        continue;

      uint32_t cls = static_cast<uint32_t>(classes[idx]);
      entry.confidence = scores[idx] * 100.0f;
      entry.name  = labels_parser_.GetLabel(cls);
      entry.color = labels_parser_.GetColor(cls);
      if (entry.name == "unknown")
        continue;

      LOG(logger_, kLog, "gearguard: %s %.1f Box[%f %f %f %f]",
          entry.name.c_str(), entry.confidence,
          entry.top, entry.left, entry.bottom, entry.right);

      int32_t nms = NonMaxSuppression(entry, detections);
      if (nms == -2)
        continue;
      if (nms >= 0)
        detections.erase(detections.begin() + nms);
      detections.emplace_back(std::move(entry));
    }
    return true;
  }

 private:
  void TransformDimensions(ObjectDetection& box, const Region& region) {
    box.top    = (box.top    - region.y) / region.height;
    box.bottom = (box.bottom - region.y) / region.height;
    box.left   = (box.left   - region.x) / region.width;
    box.right  = (box.right  - region.x) / region.width;
  }

  float IntersectionScore(const ObjectDetection& l, const ObjectDetection& r) {
    float xA = std::max(l.left, r.left);
    float yA = std::max(l.top, r.top);
    float xB = std::min(l.right, r.right);
    float yB = std::min(l.bottom, r.bottom);
    float inter = std::max(0.0f, xB - xA) * std::max(0.0f, yB - yA);
    float la = (l.right - l.left) * (l.bottom - l.top);
    float ra = (r.right - r.left) * (r.bottom - r.top);
    return inter / (la + ra - inter + 1e-5f);
  }

  int32_t NonMaxSuppression(const ObjectDetection& l, const ObjectDetections& boxes) {
    for (uint32_t i = 0; i < boxes.size(); i++) {
      const ObjectDetection& r = boxes[i];
      if (l.name != r.name)
        continue;
      if (IntersectionScore(l, r) <= kNMSIntersectionThreshold)
        continue;
      if (l.confidence > r.confidence)
        return i;
      if (l.confidence <= r.confidence)
        return -2;
    }
    return -1;
  }

  LogCallback  logger_;
  double       threshold_;
  LabelsParser labels_parser_;
};

extern "C" IModule* NewModule(LogCallback logger) {
  return new Module(logger);
}
