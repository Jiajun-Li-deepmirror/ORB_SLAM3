/**
 * Standalone visual place recognition (VPR) evaluation on the Nordland dataset: compares DBoW2
 * (ORB bag-of-words) against CosPlace (see PlaceRecognizer.h) at the actual task both are used
 * for in ORB-SLAM3 -- given a query image, retrieve the best-matching keyframe from a database
 * built from a *different* pass of the same route. Nordland's winter/summer folders are
 * pre-aligned so image N in one folder and image N in the other are (close to) the same
 * physical location on the railway -- see https://huggingface.co/datasets/Somayeh-h/Nordland.
 *
 * This does NOT run full SLAM: Nordland has no depth/stereo and is only sparsely, non-uniformly
 * sampled from the original video, so there's no meaningful continuous visual odometry to run.
 * The task DBoW2/CosPlace are actually responsible for -- "does this image match one we've seen
 * before" -- is exactly a single-image retrieval problem, so that's what's measured directly:
 * Recall@1 and Recall@5 over the ids common to both the winter and summer image samples.
 *
 * Usage: ./nordland_vpr_eval path_to_vocabulary path_to_cosplace_onnx winter_dir summer_dir [use_cuda]
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <regex>

#include "ORBextractor.h"
#include "ORBVocabulary.h"
#include "Converter.h"
#include "PlaceRecognizer.h"

using namespace std;
using namespace ORB_SLAM3;

// Maps image id -> full file path, for every "images-XXXXX.png" file in dir.
map<int, string> listImagesById(const string &dir)
{
    map<int, string> result;
    DIR *d = opendir(dir.c_str());
    if(!d)
    {
        cerr << "Could not open directory: " << dir << endl;
        return result;
    }
    std::regex re("images-([0-9]+)\\.png");
    struct dirent *entry;
    while((entry = readdir(d)) != nullptr)
    {
        string name = entry->d_name;
        std::smatch m;
        if(std::regex_match(name, m, re))
        {
            int id = std::stoi(m[1].str());
            result[id] = dir + "/" + name;
        }
    }
    closedir(d);
    return result;
}

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        cerr << "Usage: ./nordland_vpr_eval path_to_vocabulary path_to_cosplace_onnx winter_dir summer_dir [use_cuda]" << endl;
        return 1;
    }
    string vocPath = argv[1];
    string onnxPath = argv[2];
    string winterDir = argv[3];
    string summerDir = argv[4];
    bool useCuda = argc > 5 && string(argv[5]) == "1";

    cout << "Loading ORB vocabulary from " << vocPath << " ..." << endl;
    ORBVocabulary voc;
    if(!voc.loadFromTextFile(vocPath))
    {
        cerr << "Failed to load vocabulary." << endl;
        return 1;
    }
    cout << "Vocabulary loaded." << endl;

    PlaceRecognizer placeRecognizer(onnxPath, useCuda, 640, 480);
    if(!placeRecognizer.isEnabled())
    {
        cerr << "PlaceRecognizer failed to load CosPlace model, aborting." << endl;
        return 1;
    }

    map<int, string> winterFiles = listImagesById(winterDir);
    map<int, string> summerFiles = listImagesById(summerDir);
    cout << "winter images: " << winterFiles.size() << ", summer images: " << summerFiles.size() << endl;

    vector<int> commonIds;
    for(auto &p : winterFiles)
        if(summerFiles.count(p.first))
            commonIds.push_back(p.first);
    sort(commonIds.begin(), commonIds.end());
    cout << "common ids (usable for evaluation): " << commonIds.size() << endl;
    if(commonIds.size() < 20)
    {
        cerr << "Too few common ids to get a meaningful Recall number, aborting." << endl;
        return 1;
    }

    ORBextractor orb(2000, 1.2f, 8, 20, 7);

    // Build the "winter" database: for every common id, ORB BoW vector + CosPlace descriptor.
    struct DBEntry { int id; DBoW2::BowVector bow; vector<float> cosplace; };
    vector<DBEntry> database;
    database.reserve(commonIds.size());

    vector<double> cosplaceTimesMs;

    cout << "Building winter database (" << commonIds.size() << " images)..." << endl;
    for(size_t i = 0; i < commonIds.size(); i++)
    {
        int id = commonIds[i];
        cv::Mat im = cv::imread(winterFiles[id], cv::IMREAD_COLOR);
        if(im.empty()) continue;

        vector<cv::KeyPoint> kps;
        cv::Mat desc;
        vector<int> vLapping = {0,0};
        orb(im, cv::Mat(), kps, desc, vLapping);

        DBEntry e;
        e.id = id;
        if(!desc.empty())
        {
            DBoW2::FeatureVector fv;
            voc.transform(Converter::toDescriptorVector(desc), e.bow, fv, 4);
        }
        e.cosplace = placeRecognizer.computeDescriptor(im);
        cosplaceTimesMs.push_back(placeRecognizer.GetLastInferenceMs());
        database.push_back(e);

        if(i % 200 == 0)
            cout << "  " << i << "/" << commonIds.size() << endl;
    }
    cout << "Database built: " << database.size() << " entries." << endl;

    // Query with the "summer" image at each common id, see if the correct winter id comes back.
    int dbowRecallAt1 = 0, dbowRecallAt5 = 0;
    int cosplaceRecallAt1 = 0, cosplaceRecallAt5 = 0;
    int nQueries = 0;

    cout << "Querying with summer images..." << endl;
    for(size_t qi = 0; qi < commonIds.size(); qi++)
    {
        int queryId = commonIds[qi];
        cv::Mat im = cv::imread(summerFiles[queryId], cv::IMREAD_COLOR);
        if(im.empty()) continue;

        vector<cv::KeyPoint> kps;
        cv::Mat desc;
        vector<int> vLapping = {0,0};
        orb(im, cv::Mat(), kps, desc, vLapping);
        if(desc.empty())
            continue;

        DBoW2::BowVector queryBow;
        DBoW2::FeatureVector queryFv;
        voc.transform(Converter::toDescriptorVector(desc), queryBow, queryFv, 4);
        vector<float> queryCosplace = placeRecognizer.computeDescriptor(im);
        cosplaceTimesMs.push_back(placeRecognizer.GetLastInferenceMs());

        // DBoW2: score against every database entry, rank descending.
        vector<pair<float,int> > dbowScores;
        dbowScores.reserve(database.size());
        for(const DBEntry &e : database)
            dbowScores.emplace_back(voc.score(queryBow, e.bow), e.id);
        sort(dbowScores.begin(), dbowScores.end(), [](const pair<float,int>&a, const pair<float,int>&b){ return a.first > b.first; });

        // CosPlace: cosine similarity (descriptors are already L2-normalised) against every entry.
        vector<pair<float,int> > cosplaceScores;
        cosplaceScores.reserve(database.size());
        for(const DBEntry &e : database)
        {
            float dot = 0.f;
            for(size_t j = 0; j < e.cosplace.size() && j < queryCosplace.size(); j++)
                dot += e.cosplace[j]*queryCosplace[j];
            cosplaceScores.emplace_back(dot, e.id);
        }
        sort(cosplaceScores.begin(), cosplaceScores.end(), [](const pair<float,int>&a, const pair<float,int>&b){ return a.first > b.first; });

        nQueries++;
        if(!dbowScores.empty() && dbowScores[0].second == queryId) dbowRecallAt1++;
        for(int k = 0; k < min<int>(5, dbowScores.size()); k++)
            if(dbowScores[k].second == queryId) { dbowRecallAt5++; break; }

        if(!cosplaceScores.empty() && cosplaceScores[0].second == queryId) cosplaceRecallAt1++;
        for(int k = 0; k < min<int>(5, cosplaceScores.size()); k++)
            if(cosplaceScores[k].second == queryId) { cosplaceRecallAt5++; break; }

        if(qi % 200 == 0)
            cout << "  query " << qi << "/" << commonIds.size() << endl;
    }

    cout << endl << "===================== RESULTS =====================" << endl;
    cout << "Queries evaluated: " << nQueries << endl;
    cout << "DBoW2    Recall@1: " << (100.0*dbowRecallAt1/nQueries) << "%  Recall@5: " << (100.0*dbowRecallAt5/nQueries) << "%" << endl;
    cout << "CosPlace Recall@1: " << (100.0*cosplaceRecallAt1/nQueries) << "%  Recall@5: " << (100.0*cosplaceRecallAt5/nQueries) << "%" << endl;

    if(!cosplaceTimesMs.empty())
    {
        // Drop the first (chronologically earliest) call: CUDA EP pays a one-time
        // context/kernel-compile cost on its first Run(), which would skew steady-state stats.
        vector<double> steadyState(cosplaceTimesMs.begin() + 1, cosplaceTimesMs.end());
        sort(steadyState.begin(), steadyState.end());
        double sum = accumulate(steadyState.begin(), steadyState.end(), 0.0);
        double mean = sum / steadyState.size();
        double median = steadyState[steadyState.size()/2];
        double p95 = steadyState[(size_t)(steadyState.size()*0.95)];

        cout << endl << "CosPlace per-call inference time (Ort::Session::Run() only, n=" << cosplaceTimesMs.size()
             << ", excluding first warm-up call):" << endl;
        cout << "  first call (warm-up): " << cosplaceTimesMs.front() << " ms" << endl;
        cout << "  mean:   " << mean << " ms" << endl;
        cout << "  median: " << median << " ms" << endl;
        cout << "  min:    " << steadyState.front() << " ms" << endl;
        cout << "  p95:    " << p95 << " ms" << endl;
        cout << "  max:    " << steadyState.back() << " ms" << endl;
    }

    return 0;
}
