#include "DynamicDetector.h"
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace ORB_SLAM3
{

DynamicDetector::DynamicDetector(const std::string &onnxModelPath, float confThreshold,
                                  float nmsThreshold, float depthEpsilon,
                                  const std::vector<int> &dynamicClassIds)
    : mbEnabled(false), mConfThreshold(confThreshold), mNmsThreshold(nmsThreshold),
      mDepthEpsilon(depthEpsilon), mDynamicClassIds(dynamicClassIds)
{
    if(onnxModelPath.empty())
    {
        std::cout << "[DynamicDetector] No ONNX model path given, dynamic-object masking is disabled." << std::endl;
        return;
    }

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
                  << " (conf=" << mConfThreshold << ", nms=" << mNmsThreshold
                  << ", depthEpsilon=" << mDepthEpsilon << "m, "
                  << mDynamicClassIds.size() << " dynamic class id(s))" << std::endl;
    }
    catch(const cv::Exception &e)
    {
        std::cerr << "[DynamicDetector] Exception loading ONNX model: " << e.what() << std::endl;
        mbEnabled = false;
    }
}

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
    mNet.setInput(blob);

    cv::Mat output = mNet.forward();
    // Output shape is [1, numAnchors, 5+numClasses] (e.g. [1,25200,85]): each row is already
    // sigmoid-activated and decoded to [cx, cy, w, h, objectness, class0..classN] in pixels of
    // the INPUT_SIZE x INPUT_SIZE input image (this is the pre-2022 YOLOv5 "Detect" head format,
    // chosen instead of YOLOv8's anchor-free head because OpenCV 4.5.4's ONNX importer cannot
    // parse the newer head -- see the AVX/-mno-avx512f-style compatibility notes in this repo's
    // commit history for the general pattern of "old OpenCV, new export" mismatches).
    const int numAnchors = output.size[1];
    const int dims = output.size[2];
    cv::Mat outMat(numAnchors, dims, CV_32F, output.ptr<float>());

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
