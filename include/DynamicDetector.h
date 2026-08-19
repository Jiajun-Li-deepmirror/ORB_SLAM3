/**
 * DynamicDetector: object-detection + depth-threshold dynamic feature masking,
 * porting the core idea of Dynamic-VINS (Liu et al., RA-L 2022) into ORB-SLAM3's
 * RGB-D pipeline.
 *
 * Pipeline: YOLOv5 with the classic (pre-anchor-free) Detect head, ONNX via cv::dnn, detects
 * candidate dynamic-object boxes
 * (person by default) on the RGB image. For each box, the depth image is used
 * to build a per-pixel dynamic mask without needing real semantic segmentation:
 * the box's farthest corner depth is assumed to be background, and pixels
 * closer than a threshold derived from (center depth, farthest-corner depth,
 * a fixed epsilon) are marked dynamic. See detectDynamicMask() for the exact
 * rule (paper eq. 1-2).
 */
#ifndef DYNAMICDETECTOR_H
#define DYNAMICDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

#ifdef WITH_ORT_CUDA
#include <onnxruntime_cxx_api.h>
#endif

namespace ORB_SLAM3
{

class DynamicDetector
{
public:
    struct Detection
    {
        cv::Rect box;
        int classId;
        float confidence;
    };

    // onnxModelPath: path to a classic-head YOLOv5 .onnx export (640x640 input, [1,25200,85]
    // output -- OpenCV 4.5.4's ONNX importer cannot parse YOLOv8/anchor-free exports, see
    // models/README.md for how this file was produced).
    // dynamicClassIds: COCO class ids treated as potentially dynamic (0 = person).
    // depthEpsilon: predefined distance (metres) used in the depth-threshold rule,
    // sized to the typical depth extent of the targeted dynamic objects.
    // useCuda: run YOLO inference through ONNX Runtime's CUDA execution provider instead of
    // OpenCV's CPU-only DNN module. Only has an effect if built WITH_ORT_CUDA; silently falls
    // back to the OpenCV CPU path otherwise so callers don't need an #ifdef at the call site.
    DynamicDetector(const std::string &onnxModelPath, float confThreshold = 0.35f,
                     float nmsThreshold = 0.45f, float depthEpsilon = 0.4f,
                     const std::vector<int> &dynamicClassIds = std::vector<int>{0},
                     bool useCuda = false);
    ~DynamicDetector();

    // imRGB: colour or grayscale image, any size. imDepth: CV_32F depth in metres,
    // 0 = invalid/unknown, same size as imRGB. Returns a CV_8U mask the same size
    // as imRGB (255 = dynamic pixel, 0 = static/unknown).
    cv::Mat detectDynamicMask(const cv::Mat &imRGB, const cv::Mat &imDepth);

    bool isEnabled() const { return mbEnabled; }

    // Last frame's raw detections (all classes, post-NMS), kept for debugging/visualisation.
    const std::vector<Detection>& lastDetections() const { return mvLastDetections; }

private:
    std::vector<Detection> runYolo(const cv::Mat &imRGB);
    // Runs one forward pass and returns the raw [numAnchors x dims] output as a CV_32F cv::Mat,
    // regardless of which backend actually executed it -- runYolo()'s postprocessing (objectness
    // * class score, box decode, NMS) is identical either way.
    cv::Mat forward(const cv::Mat &blob);
    void applyDepthThresholdMask(const cv::Mat &imDepth, const cv::Rect &box, cv::Mat &mask) const;

    cv::dnn::Net mNet;
    bool mbEnabled;
    bool mbUseCuda;
    float mConfThreshold;
    float mNmsThreshold;
    float mDepthEpsilon;
    std::vector<int> mDynamicClassIds;
    static const int INPUT_SIZE = 640;

    std::vector<Detection> mvLastDetections;

#ifdef WITH_ORT_CUDA
    std::unique_ptr<Ort::Env> mpOrtEnv;
    std::unique_ptr<Ort::Session> mpOrtSession;
    std::string mOrtInputName;
    std::string mOrtOutputName;
#endif
};

} // namespace ORB_SLAM3

#endif // DYNAMICDETECTOR_H
