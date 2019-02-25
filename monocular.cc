#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<iomanip>

#include<opencv2/core/core.hpp>

#include"System.h"

using namespace std;

int main(int argc, char **argv)
{
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM("/home/victor/datasets/ORBvoc.txt", "/home/victor/datasets/KITTI04-12.yaml", ORB_SLAM2::System::MONOCULAR, true);

    cv::VideoCapture capture(1);

    // Main loop
    cv::Mat im;

    std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

    do
    {
        // Read image from file
        capture >> im;

        if(im.empty())
        {
            cerr << endl << "Failed to load image\n" << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        double tframe = std::chrono::duration_cast<std::chrono::duration<double> >(currentTime - time).count();
        time = currentTime;

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe);

        usleep(30);
    } while(!im.empty());

    // Stop all threads
    SLAM.Shutdown();

    return 0;

}

