#include "PlaceRecognizer.h"
#include "KeyFrame.h"
#include "Map.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>

namespace ORB_SLAM3
{

PlaceRecognizer::PlaceRecognizer(const std::string &onnxModelPath, bool useCuda,
                                  int inputWidth, int inputHeight)
    : mbEnabled(false), mbUseCuda(false), mInputWidth(inputWidth), mInputHeight(inputHeight)
{
    if(onnxModelPath.empty())
    {
        std::cout << "[PlaceRecognizer] No ONNX model path given, CosPlace-based candidate "
                      "detection is disabled." << std::endl;
        return;
    }

#ifdef WITH_ORT_CUDA
    try
    {
        mpOrtEnv.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "PlaceRecognizer"));
        Ort::SessionOptions options;
        if(useCuda)
        {
            OrtCUDAProviderOptions cudaOptions{};
            options.AppendExecutionProvider_CUDA(cudaOptions);
        }
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        mpOrtSession.reset(new Ort::Session(*mpOrtEnv, onnxModelPath.c_str(), options));

        Ort::AllocatorWithDefaultOptions allocator;
        mOrtInputName = mpOrtSession->GetInputNameAllocated(0, allocator).get();
        mOrtOutputName = mpOrtSession->GetOutputNameAllocated(0, allocator).get();

        mbEnabled = true;
        mbUseCuda = useCuda;
        std::cout << "[PlaceRecognizer] Loaded CosPlace model from " << onnxModelPath
                  << " via ONNX Runtime " << (useCuda ? "CUDA EP" : "CPU") << std::endl;

        mWorkerThread = std::thread(&PlaceRecognizer::Run, this);
    }
    catch(const Ort::Exception &e)
    {
        std::cerr << "[PlaceRecognizer] Failed to load CosPlace model (" << e.what()
                  << "); CosPlace-based candidate detection is disabled." << std::endl;
        mpOrtSession.reset();
        mpOrtEnv.reset();
        mbEnabled = false;
    }
#else
    std::cerr << "[PlaceRecognizer] Built without WITH_ORT_CUDA (ONNX Runtime not found at "
                  "configure time); CosPlace-based candidate detection is disabled regardless "
                  "of Detector.OnnxPath, since OpenCV's DNN module can't parse this model "
                  "(unsupported GeM-pooling 'Reciprocal' op)." << std::endl;
#endif
}

PlaceRecognizer::~PlaceRecognizer()
{
    RequestFinish();
}

void PlaceRecognizer::Run()
{
    while(true)
    {
        std::pair<KeyFrame*, cv::Mat> job;
        {
            std::unique_lock<std::mutex> lock(mMutexQueue);
            mCondQueue.wait(lock, [this]{ return !mJobQueue.empty() || mbFinishRequested; });
            if(mJobQueue.empty() && mbFinishRequested)
                return;
            job = mJobQueue.front();
            mJobQueue.pop();
        }
        std::vector<float> descriptor = computeDescriptor(job.second);
        addKeyFrame(job.first, descriptor);
    }
}

void PlaceRecognizer::RequestKeyFrame(KeyFrame *pKF, const cv::Mat &im)
{
    if(!mbEnabled)
        return;
    std::unique_lock<std::mutex> lock(mMutexQueue);
    mJobQueue.emplace(pKF, im.clone());
    mCondQueue.notify_one();
}

void PlaceRecognizer::RequestFinish()
{
    {
        std::unique_lock<std::mutex> lock(mMutexQueue);
        mbFinishRequested = true;
    }
    mCondQueue.notify_all();
    if(mWorkerThread.joinable())
        mWorkerThread.join();
}

cv::Mat PlaceRecognizer::forward(const cv::Mat &blob)
{
#ifdef WITH_ORT_CUDA
    CV_Assert(blob.isContinuous());
    std::array<int64_t, 4> inputShape{1, 3, mInputHeight, mInputWidth};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, const_cast<float*>(blob.ptr<float>()), blob.total(),
        inputShape.data(), inputShape.size());

    const char *inputNames[] = {mOrtInputName.c_str()};
    const char *outputNames[] = {mOrtOutputName.c_str()};
    auto t0 = std::chrono::steady_clock::now();
    auto outputTensors = mpOrtSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                                            outputNames, 1);
    auto t1 = std::chrono::steady_clock::now();
    mLastInferenceMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const int dim = static_cast<int>(outputTensors.front().GetTensorTypeAndShapeInfo().GetShape()[1]);
    const float *data = outputTensors.front().GetTensorData<float>();
    return cv::Mat(1, dim, CV_32F, const_cast<float*>(data)).clone();
#else
    (void)blob;
    return cv::Mat();
#endif
}

std::vector<float> PlaceRecognizer::computeDescriptor(const cv::Mat &im)
{
    if(!mbEnabled)
        return std::vector<float>();

    // CosPlace was trained on RGB ImageNet-normalised crops; blobFromImage's mean-subtraction
    // only takes a per-channel scalar, which is close enough to ImageNet normalisation here --
    // exact renormalisation isn't critical since we only ever compare descriptors to each other
    // (cosine similarity), not to some external reference.
    //
    // Callers (Tracking::CreateNewKeyFrame/Relocalization) pass mImGray, which ORB-SLAM3 keeps
    // single-channel for every sensor type including monocular/stereo grayscale datasets like
    // KITTI -- the ONNX graph's input shape is hardcoded to 3 channels in forward(), so a
    // grayscale mat must be expanded first or ONNX Runtime throws a shape-mismatch exception.
    cv::Mat imColor;
    if(im.channels() == 1)
        cv::cvtColor(im, imColor, cv::COLOR_GRAY2BGR);
    else
        imColor = im;

    cv::Mat blob;
    cv::dnn::blobFromImage(imColor, blob, 1.0/255.0, cv::Size(mInputWidth, mInputHeight),
                            cv::Scalar(), true, false);

    cv::Mat out = forward(blob);
    if(out.empty())
        return std::vector<float>();

    std::vector<float> descriptor(out.ptr<float>(), out.ptr<float>() + out.total());

    // L2-normalise so cosine similarity reduces to a plain dot product in findTopK().
    float norm = 0.f;
    for(float v : descriptor) norm += v*v;
    norm = std::sqrt(norm);
    if(norm > 1e-12f)
        for(float &v : descriptor) v /= norm;

    return descriptor;
}

void PlaceRecognizer::addKeyFrame(KeyFrame *pKF, const std::vector<float> &descriptor)
{
    if(!mbEnabled || descriptor.empty())
        return;
    std::unique_lock<std::mutex> lock(mMutexDB);
    mvpKeyFrames.push_back(pKF);
    mvDescriptors.push_back(descriptor);
}

std::vector<float> PlaceRecognizer::getDescriptor(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexDB);
    for(size_t i = 0; i < mvpKeyFrames.size(); i++)
        if(mvpKeyFrames[i] == pKF)
            return mvDescriptors[i];
    return std::vector<float>();
}

void PlaceRecognizer::eraseKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexDB);
    for(size_t i = 0; i < mvpKeyFrames.size(); i++)
    {
        if(mvpKeyFrames[i] == pKF)
        {
            // Swap-and-pop: order doesn't matter, this is O(1) instead of O(n) for a vector erase.
            mvpKeyFrames[i] = mvpKeyFrames.back();
            mvDescriptors[i] = mvDescriptors.back();
            mvpKeyFrames.pop_back();
            mvDescriptors.pop_back();
            return;
        }
    }
}

std::vector<KeyFrame*> PlaceRecognizer::findTopK(const std::vector<float> &queryDesc, int k,
                                                  const std::set<KeyFrame*> &vExclude)
{
    std::vector<KeyFrame*> result;
    if(!mbEnabled || queryDesc.empty())
        return result;

    std::vector<std::pair<float,KeyFrame*> > scored;
    {
        std::unique_lock<std::mutex> lock(mMutexDB);
        scored.reserve(mvpKeyFrames.size());
        for(size_t i = 0; i < mvpKeyFrames.size(); i++)
        {
            KeyFrame *pKFi = mvpKeyFrames[i];
            if(vExclude.count(pKFi))
                continue;
            if(pKFi->isBad())
                continue;
            if(pKFi->GetMap() && pKFi->GetMap()->IsBad())
                continue;

            const std::vector<float> &d = mvDescriptors[i];
            float dot = 0.f;
            const size_t n = std::min(d.size(), queryDesc.size());
            for(size_t j = 0; j < n; j++)
                dot += d[j] * queryDesc[j];
            scored.emplace_back(dot, pKFi);
        }
    }

    std::sort(scored.begin(), scored.end(),
              [](const std::pair<float,KeyFrame*> &a, const std::pair<float,KeyFrame*> &b) {
                  return a.first > b.first;
              });

    result.reserve(std::min<size_t>(k, scored.size()));
    for(size_t i = 0; i < scored.size() && static_cast<int>(result.size()) < k; i++)
        result.push_back(scored[i].second);

    return result;
}

} // namespace ORB_SLAM3
