/**
 * Same as rgbd_tum.cc, but switches to pure localization mode (System::ActivateLocalizationMode)
 * right after loading the map, instead of continuing to map. Used to check whether dynamic-object
 * masking (DynamicDetector, see Detector.OnnxPath in the settings file) is still needed once
 * tracking is done against an already-built, dynamic-object-free map: with no new keyframes/map
 * points being created, dynamic-object keypoints simply have no map point to match against and
 * get naturally rejected, without needing a per-frame YOLO pass.
 *
 * Usage is identical to rgbd_tum, but the settings file is expected to set
 * System.LoadAtlasFromFile to a map built beforehand (e.g. with rgbd_tum + Detector.OnnxPath +
 * System.SaveAtlasToFile).
 */

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum_localization path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);

    // Loading a map does NOT resume tracking against it: a fresh session still bootstraps a
    // brand new (unrelated) map from scratch on frame 1, and only reconnects to the loaded map
    // once place recognition merges the two. Activating localization mode from frame 0 disables
    // LocalMapping before that bootstrap map can sustain itself (no new keyframes allowed), so
    // tracking is lost almost immediately and never gets the chance to merge. Run a short warm-up
    // in normal SLAM mode first so the merge can happen, then switch to localization-only.
    const int nWarmupFrames = 30;

    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    vector<float> vTimesTrackLocOnly; // tracking time for frames after the warm-up, i.e. once
                                       // actually running in pure localization mode

    cout << endl << "-------" << endl;
    cout << "Start processing sequence in LOCALIZATION-ONLY mode ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        if(ni == nWarmupFrames)
        {
            cout << "Switching to localization-only mode after " << nWarmupFrames << " warm-up frames" << endl;
            SLAM.ActivateLocalizationMode();
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        SLAM.TrackRGBD(imRGB,imD,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni]=ttrack;
        if(ni >= nWarmupFrames)
            vTimesTrackLocOnly.push_back(ttrack);

        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    SLAM.Shutdown();

    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
        totaltime+=vTimesTrack[ni];
    cout << "-------" << endl << endl;
    cout << "median tracking time (all frames, incl. warm-up): " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time (all frames, incl. warm-up): " << totaltime/nImages << endl;

    if(!vTimesTrackLocOnly.empty())
    {
        sort(vTimesTrackLocOnly.begin(), vTimesTrackLocOnly.end());
        float locTotal = 0;
        for(float t : vTimesTrackLocOnly) locTotal += t;
        cout << "median tracking time (localization-only phase): " << vTimesTrackLocOnly[vTimesTrackLocOnly.size()/2] << endl;
        cout << "mean tracking time (localization-only phase): " << locTotal/vTimesTrackLocOnly.size() << endl;
    }

    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}
