#include "DynamicDetector.h"
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <array>

namespace ORB_SLAM3
{

DynamicDetector::DynamicDetector(const std::string &onnxModelPath, float confThreshold,
                                  float nmsThreshold, float depthEpsilon,
                                  const std::vector<int> &dynamicClassIds, bool useCuda)
    : mbEnabled(false), mbUseCuda(false), mConfThreshold(confThreshold), mNmsThreshold(nmsThreshold),
      mDepthEpsilon(depthEpsilon), mDynamicClassIds(dynamicClassIds)
{
    if(onnxModelPath.empty())
    {
        std::cout << "[DynamicDetector] No ONNX model path given, dynamic-object masking is disabled." << std::endl;
        return;
    }

#ifdef WITH_ORT_CUDA
    if(useCuda)
    {
        try
        {
            mpOrtEnv.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DynamicDetector"));
            Ort::SessionOptions options;
            OrtCUDAProviderOptions cudaOptions{};
            options.AppendExecutionProvider_CUDA(cudaOptions);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            mpOrtSession.reset(new Ort::Session(*mpOrtEnv, onnxModelPath.c_str(), options));

            Ort::AllocatorWithDefaultOptions allocator;
            mOrtInputName = mpOrtSession->GetInputNameAllocated(0, allocator).get();
            mOrtOutputName = mpOrtSession->GetOutputNameAllocated(0, allocator).get();

            mbEnabled = true;
            mbUseCuda = true;
            std::cout << "[DynamicDetector] Loaded YOLO model from " << onnxModelPath
                      << " via ONNX Runtime CUDA EP (conf=" << mConfThreshold << ", nms=" << mNmsThreshold
                      << ", depthEpsilon=" << mDepthEpsilon << "m, "
                      << mDynamicClassIds.size() << " dynamic class id(s))" << std::endl;
            return;
        }
        catch(const Ort::Exception &e)
        {
            std::cerr << "[DynamicDetector] Failed to init ONNX Runtime CUDA EP (" << e.what()
                      << "), falling back to OpenCV CPU backend." << std::endl;
            mpOrtSession.reset();
            mpOrtEnv.reset();
        }
    }
#else
    if(useCuda)
        std::cerr << "[DynamicDetector] Built without WITH_ORT_CUDA, ignoring useCuda=true and "
                      "falling back to OpenCV CPU backend." << std::endl;
#endif

    try
    {
        mNet = cv::dnn::readNetFromONNX(onnxModelPath);
        if(mNet.empty())
        {
            std::cerr << "[DynamicDetector] Failed to load ONNX model at: " << onnxModelPath << std::endl;
            return;
        }
        mNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        mNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        mbEnabled = true;
        std::cout << "[DynamicDetector] Loaded YOLO model from " << onnxModelPath
                  << " via OpenCV CPU backend (conf=" << mConfThreshold << ", nms=" << mNmsThreshold
                  << ", depthEpsilon=" << mDepthEpsilon << "m, "
                  << mDynamicClassIds.size() << " dynamic class id(s))" << std::endl;
    }
    catch(const cv::Exception &e)
    {
        std::cerr << "[DynamicDetector] Exception loading ONNX model: " << e.what() << std::endl;
        mbEnabled = false;
    }
}

DynamicDetector::~DynamicDetector() = default;

std::vector<DynamicDetector::Detection> DynamicDetector::runYolo(const cv::Mat &imRGB)
{
    std::vector<Detection> detections;
    if(!mbEnabled)
        return detections;

    cv::Mat blob;
    // Classic-head YOLOv5 ONNX export expects RGB, [0,1]-normalised, NCHW, INPUT_SIZE x
    // INPUT_SIZE input. We resize directly to a square (ignoring aspect ratio) rather than
    // letterboxing, which is simple and accurate enough for the person-sized boxes this
    // module targets.
    cv::dnn::blobFromImage(imRGB, blob, 1.0/255.0, cv::Size(INPUT_SIZE, INPUT_SIZE),
                            cv::Scalar(), true, false);

    // Output shape is [1, numAnchors, 5+numClasses] (e.g. [1,25200,85]): each row is already
    // sigmoid-activated and decoded to [cx, cy, w, h, objectness, class0..classN] in pixels of
    // the INPUT_SIZE x INPUT_SIZE input image (this is the pre-2022 YOLOv5 "Detect" head format,
    // chosen instead of YOLOv8's anchor-free head because OpenCV 4.5.4's ONNX importer cannot
    // parse the newer head -- see the AVX/-mno-avx512f-style compatibility notes in this repo's
    // commit history for the general pattern of "old OpenCV, new export" mismatches).
    cv::Mat outMat = forward(blob);
    const int numAnchors = outMat.rows;
    const int dims = outMat.cols;

    const float scaleX = static_cast<float>(imRGB.cols) / INPUT_SIZE;
    const float scaleY = static_cast<float>(imRGB.rows) / INPUT_SIZE;
    const int numClasses = dims - 5;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for(int i = 0; i < numAnchors; i++)
    {
        const float *row = outMat.ptr<float>(i);
        const float cx = row[0], cy = row[1], w = row[2], h = row[3];
        const float objectness = row[4];
        if(objectness < mConfThreshold)
            continue;

        int bestClass = -1;
        float bestClassScore = 0.f;
        for(int c = 0; c < numClasses; c++)
        {
            if(row[5+c] > bestClassScore)
            {
                bestClassScore = row[5+c];
                bestClass = c;
            }
        }

        const float score = objectness * bestClassScore; // standard YOLOv5 postprocessing
        if(score < mConfThreshold || bestClass < 0)
            continue;
        if(std::find(mDynamicClassIds.begin(), mDynamicClassIds.end(), bestClass) == mDynamicClassIds.end())
            continue;

        const int x = cvRound((cx - w/2.f) * scaleX);
        const int y = cvRound((cy - h/2.f) * scaleY);
        const int bw = cvRound(w * scaleX);
        const int bh = cvRound(h * scaleY);

        boxes.emplace_back(x, y, bw, bh);
        scores.push_back(score);
        classIds.push_back(bestClass);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, mConfThreshold, mNmsThreshold, keep);

    detections.reserve(keep.size());
    const cv::Rect imgBounds(0, 0, imRGB.cols, imRGB.rows);
    for(int idx : keep)
    {
        cv::Rect box = boxes[idx] & imgBounds; // clamp to image
        if(box.width <= 0 || box.height <= 0)
            continue;
        detections.push_back({box, classIds[idx], scores[idx]});
    }

    return detections;
}

cv::Mat DynamicDetector::forward(const cv::Mat &blob)
{
#ifdef WITH_ORT_CUDA
    if(mbUseCuda)
    {
        CV_Assert(blob.isContinuous());
        std::array<int64_t, 4> inputShape{1, 3, INPUT_SIZE, INPUT_SIZE};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(blob.ptr<float>()), blob.total(),
            inputShape.data(), inputShape.size());

        const char *inputNames[] = {mOrtInputName.c_str()};
        const char *outputNames[] = {mOrtOutputName.c_str()};
        auto outputTensors = mpOrtSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                                outputNames, 1);

        const Ort::TensorTypeAndShapeInfo shapeInfo = outputTensors.front().GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> shape = shapeInfo.GetShape(); // [1, numAnchors, dims]
        const int numAnchors = static_cast<int>(shape[1]);
        const int dims = static_cast<int>(shape[2]);
        const float *data = outputTensors.front().GetTensorData<float>();
        // Copy out: the Ort::Value (and the memory `data` points into) is destroyed together
        // with outputTensors when this function returns.
        return cv::Mat(numAnchors, dims, CV_32F, const_cast<float*>(data)).clone();
    }
#endif
    mNet.setInput(blob);
    cv::Mat output = mNet.forward();
    return cv::Mat(output.size[1], output.size[2], CV_32F, output.ptr<float>()).clone();
}

void DynamicDetector::applyDepthThresholdMask(const cv::Mat &imDepth, const cv::Rect &box, cv::Mat &mask) const
{
    // Dynamic-VINS depth-threshold rule (Liu et al., RA-L 2022, eq. 1-2): the box's farthest
    // corner is assumed to lie on the (static) background, the box centre is assumed to lie on
    // the (dynamic) foreground object. Pixels closer than a threshold between the two are
    // classified as dynamic; the exact threshold placement depends on how large the gap is.
    auto depthAt = [&](int x, int y) -> float {
        x = std::min(std::max(x, 0), imDepth.cols - 1);
        y = std::min(std::max(y, 0), imDepth.rows - 1);
        return imDepth.at<float>(y, x);
    };

    const float dtl = depthAt(box.x, box.y);
    const float dtr = depthAt(box.x + box.width - 1, box.y);
    const float dbl = depthAt(box.x, box.y + box.height - 1);
    const float dbr = depthAt(box.x + box.width - 1, box.y + box.height - 1);
    const float dmax = std::max(std::max(dtl, dtr), std::max(dbl, dbr));
    const float dc = depthAt(box.x + box.width/2, box.y + box.height/2);

    float dThresh;
    if(dmax > 0.f && dc > 0.f)
    {
        dThresh = (dmax - dc > mDepthEpsilon) ? 0.5f*(dmax + dc) : (dc + mDepthEpsilon);
    }
    else if(dmax > 0.f)
    {
        // Centre depth unavailable: fall back to the farthest background depth found.
        dThresh = dmax;
    }
    else
    {
        // No depth available anywhere on the box: conservatively treat the whole box as dynamic.
        dThresh = std::numeric_limits<float>::infinity();
    }

    for(int y = box.y; y < box.y + box.height; y++)
    {
        uchar *maskRow = mask.ptr<uchar>(y);
        const float *depthRow = imDepth.ptr<float>(y);
        for(int x = box.x; x < box.x + box.width; x++)
        {
            const float d = depthRow[x];
            if(std::isinf(dThresh) || (d > 0.f && d < dThresh))
                maskRow[x] = 255;
        }
    }
}

cv::Mat DynamicDetector::detectDynamicMask(const cv::Mat &imRGB, const cv::Mat &imDepth)
{
    cv::Mat mask = cv::Mat::zeros(imRGB.size(), CV_8U);
    if(!mbEnabled)
        return mask;

    mvLastDetections = runYolo(imRGB);
    for(const Detection &det : mvLastDetections)
        applyDepthThresholdMask(imDepth, det.box, mask);

    return mask;
}

} // namespace ORB_SLAM3
