/**
 * PlaceRecognizer: CosPlace-based global-descriptor place recognition, used as an additional
 * loop-closure / relocalization candidate source alongside DBoW2 (see LoopClosing.cc's call to
 * DetectNBestCandidates and Tracking::Relocalization()'s call to DetectRelocalizationCandidates).
 *
 * DBoW2's bag-of-binary-ORB-words is appearance-fragile: the same physical place, seen under a
 * different season/lighting/viewpoint, can produce a completely different word-frequency
 * histogram, so a genuine revisit is never proposed as a candidate at all. CosPlace (Berton et
 * al., CVPR 2022) encodes a whole image into a single descriptor trained specifically to be
 * similar for the same place across such appearance changes, and dissimilar for different
 * places -- see models/README.md (once added) for how the ONNX export was produced.
 *
 * This class does NOT replace DBoW2 or touch geometric verification: it only proposes
 * additional keyframe candidates by nearest-neighbour search over CosPlace descriptors. The
 * caller unions these with DBoW2's own candidates and geometric verification runs unchanged.
 *
 * Requires ONNX Runtime (see DynamicDetector.h's WITH_ORT_CUDA for the same build flag): unlike
 * the YOLO detector, no OpenCV cv::dnn fallback exists here because OpenCV 4.5.4's ONNX importer
 * can't parse CosPlace's GeM-pooling "Reciprocal" op either. Both the CPU and CUDA execution
 * providers go through ONNX Runtime; useCuda only selects which provider to attach.
 */
#ifndef PLACERECOGNIZER_H
#define PLACERECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <set>
#include <mutex>
#include <memory>
#include <thread>
#include <queue>
#include <condition_variable>
#include <utility>

#ifdef WITH_ORT_CUDA
#include <onnxruntime_cxx_api.h>
#endif

namespace ORB_SLAM3
{

class KeyFrame;

class PlaceRecognizer
{
public:
    // onnxModelPath: path to a CosPlace ONNX export (see models/README.md). Empty path disables
    // the recognizer entirely (isEnabled() == false), matching DynamicDetector's convention.
    PlaceRecognizer(const std::string &onnxModelPath, bool useCuda = false,
                     int inputWidth = 640, int inputHeight = 480);
    ~PlaceRecognizer();

    bool isEnabled() const { return mbEnabled; }

    // im: colour or grayscale image, any size (resized internally to inputWidth x inputHeight).
    // Returns an L2-normalised descriptor, or an empty vector if disabled.
    std::vector<float> computeDescriptor(const cv::Mat &im);

    // In-memory keyframe database: add when a keyframe is created, erase when it's culled --
    // mirrors KeyFrameDatabase::add()/erase()'s call sites (see Tracking::CreateNewKeyFrame()
    // and KeyFrame::SetBadFlag()).
    void addKeyFrame(KeyFrame *pKF, const std::vector<float> &descriptor);
    void eraseKeyFrame(KeyFrame *pKF);

    // Looks up the descriptor stored for pKF (computed once in Tracking::CreateNewKeyFrame()),
    // so callers that only have a KeyFrame* (no raw image) don't need to recompute it. Returns
    // an empty vector if pKF isn't in the database.
    std::vector<float> getDescriptor(KeyFrame *pKF);

    // Returns up to k keyframes whose descriptor has the highest cosine similarity to queryDesc,
    // excluding any keyframe in vExclude (the query keyframe's covisible neighbours are always
    // passed here by the caller, mirroring DetectNBestCandidates: an already-known neighbour
    // isn't a "new" loop/merge candidate) and any keyframe that GetMap()->IsBad() flags as
    // pending deletion. Results are sorted best-first.
    std::vector<KeyFrame*> findTopK(const std::vector<float> &queryDesc, int k,
                                     const std::set<KeyFrame*> &vExclude = std::set<KeyFrame*>());

    // Wall-clock time (ms) spent inside Ort::Session::Run() for the most recent computeDescriptor()
    // call -- exposed for benchmarking (CPU vs CUDA EP), not used anywhere in the SLAM pipeline itself.
    double GetLastInferenceMs() const { return mLastInferenceMs; }

    // Queues (pKF, im) for descriptor computation on a dedicated worker thread instead of blocking
    // the caller. Tracking::CreateNewKeyFrame() must use this, not computeDescriptor()+addKeyFrame()
    // directly: a synchronous call there was found to reproducibly cause KITTI tracking loss under
    // the CUDA execution provider -- the added ~4.5ms/keyframe delay shifts Tracking/LoopClosing
    // thread interleaving into DBoW2's non-thread-safe global RNG (shared, unguarded std::rand
    // state used by Sim3Solver/MLPnPSolver's RANSAC sampling -- a known "wontfix" upstream issue,
    // see github.com/UZ-SLAMLab/ORB_SLAM3/issues/71), occasionally causing RANSAC to pick a bad
    // sample set. Moving the delay off the tracking thread's critical path avoids contributing to
    // this, though it can't fix the underlying upstream non-determinism.
    // im is cloned internally since the caller's mat (e.g. mImGray) is overwritten every frame.
    void RequestKeyFrame(KeyFrame *pKF, const cv::Mat &im);

    // Stops the worker thread, blocking until its queued jobs drain. Safe to call more than once
    // (e.g. from both System::Shutdown() and the destructor); a no-op if never started.
    void RequestFinish();

private:
    cv::Mat forward(const cv::Mat &blob);
    void Run();

    bool mbEnabled;
    bool mbUseCuda;
    int mInputWidth, mInputHeight;

    double mLastInferenceMs = 0.0;

    std::thread mWorkerThread;
    std::queue<std::pair<KeyFrame*, cv::Mat> > mJobQueue;
    std::mutex mMutexQueue;
    std::condition_variable mCondQueue;
    bool mbFinishRequested = false;

    std::mutex mMutexDB;
    std::vector<KeyFrame*> mvpKeyFrames;
    std::vector<std::vector<float> > mvDescriptors;

#ifdef WITH_ORT_CUDA
    std::unique_ptr<Ort::Env> mpOrtEnv;
    std::unique_ptr<Ort::Session> mpOrtSession;
    std::string mOrtInputName;
    std::string mOrtOutputName;
#endif
};

} // namespace ORB_SLAM3

#endif // PLACERECOGNIZER_H
