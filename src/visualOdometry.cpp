#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/sfm.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>

using namespace cv;
using namespace std;

Size winSize=Size(31, 31);
TermCriteria termcrit=TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);

void featureTracking(Mat sourceFrame, Mat targetFrame, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)
{
    cv::Mat grey1;
    cv::cvtColor(sourceFrame, grey1, COLOR_BGR2GRAY);

    cv::Mat grey2;
    cv::cvtColor(targetFrame, grey2, COLOR_BGR2GRAY);

    vector<float> err;

    calcOpticalFlowPyrLK(grey1, grey2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
   // cv::cornerSubPix(grey2, points2, cv::Size(3, 3), cv::Size(), termcrit);

    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++) {
        Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0) || (pt.x<0)||(pt.y<0)) {
              if((pt.x<0) || (pt.y<0))	{
                status.at(i) = 0;
              }
              points1.erase (points1.begin() + (i - indexCorrection));
              points2.erase (points2.begin() + (i - indexCorrection));
              indexCorrection++;
        }

     }

}


void featureDetection(Mat frame, vector<Point2f>& points, std::vector<cv::Scalar> &colors)
{
    cv::Mat grey;
    cv::cvtColor(frame, grey, COLOR_BGR2GRAY);
    cv::goodFeaturesToTrack(grey, points, 1000, 0.01, 0);
    // cv::cornerSubPix(grey, points, cv::Size(3, 3), cv::Size(), termcrit);

    for (auto &i : points) {
        colors.push_back(frame.at<cv::Scalar>(i.x, i.y));
    }


}

void drawPoints(Mat frame, vector<Point2f>& points)
{
    for (auto &i : points)
        cv::circle(frame, i, 5, cv::Scalar(0, 0, 255, 255), -1);

}

void drawLines(Mat frame, vector<Point2f>& points1, vector<Point2f>& points2)
{
    for (int i = 0; i <points1.size(); ++i)
        cv::line(frame, points1[i], points2[i], cv::Scalar(0, 255, 0, 255));

}

int main( int argc, char** argv )
{
    cv::VideoCapture capture("/home/victor/testVideo/video.mp4");

    if (capture.isOpened()) {
        namedWindow( "Features", WINDOW_KEEPRATIO );
        cv::resizeWindow("Features", 800, 600);
        cv::moveWindow("Features", 80, 10);

        cv:viz::Viz3d vizWindow("Viz");
        cv::moveWindow("Viz", 500, 10);
        vizWindow.showWidget("coordSystemWidget", cv::viz::WCoordinateSystem());


        Mat camMatrix = (Mat_<double>(3,3) << 718.856, 0., 607.1928, 0.,
                         718.856, 185.2157, 0., 0., 1.);

        std::vector<double> distCoefficients = { 0, 0, 0, 0, 0 };

        cv::Vec2d focal(718.856, 718.856);
        cv::Point2d pp(607.1928, 185.2157);

        std::vector<cv::Affine3d> trajectoryPoints;

        cv::Mat captureFrame;
        cv::Mat prevFrame;
        vector<cv::Point2f> prevFeaturePoints;
        cv::Mat curFrame;
        vector<cv::Point2f> curFeaturePoints;

        int i = 0;

        cv::Mat rTotal = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat tTotal = cv::Mat::zeros(3, 1, CV_64F);

        std::vector<cv::Point3f> points;
        std::vector<cv::Vec4b> colors;

        std::vector<cv::Scalar> tempColors;

        do {
            char fileName[256];
            sprintf(fileName, "/home/victor/datasets/KITTI\ Color/00/image_2/%06d.png", i);
            captureFrame = imread(fileName);

            // capture >> captureFrame;

            if (!captureFrame.empty()) {

                cv::Mat frame;

                cv::undistort(captureFrame, frame, camMatrix, distCoefficients);

                if (prevFrame.empty()) {
                    frame.copyTo(prevFrame);
                    featureDetection(prevFrame, prevFeaturePoints, tempColors);
                    cv::Mat renderFrame;
                    prevFrame.copyTo(renderFrame);
                    drawPoints(renderFrame, prevFeaturePoints);
                    cv::imshow("Features", renderFrame);
                }
                else {
                    frame.copyTo(curFrame);
                    vector<uchar> status;
                    featureTracking(prevFrame, curFrame, prevFeaturePoints, curFeaturePoints, status);

                    if (prevFeaturePoints.size() < 500)	{
                        featureDetection(prevFrame, prevFeaturePoints, tempColors);
                        featureTracking(prevFrame, curFrame, prevFeaturePoints, curFeaturePoints, status);

                    }

                    Mat R, t;
                    cv:: Mat e;
                    e = cv::findEssentialMat(prevFeaturePoints, curFeaturePoints, focal[0], pp, RANSAC, 0.999, 1.0);

                    cv::Mat f;

                    cv::Mat prevPts(2, prevFeaturePoints.size(), CV_64F);
                    cv::Mat curPts(2, curFeaturePoints.size(), CV_64F);

                    for (auto i = 0; i < prevFeaturePoints.size(); ++i) {
                        prevPts.at<double>(0, i) = prevFeaturePoints[i].x;
                        prevPts.at<double>(1, i) = prevFeaturePoints[i].y;
                    }

                    for (auto i = 0; i < curFeaturePoints.size(); ++i) {
                        curPts.at<double>(0, i) = curFeaturePoints[i].x;
                        curPts.at<double>(1, i) = curFeaturePoints[i].y;
                    }

                    f = cv::findFundamentalMat(prevFeaturePoints, curFeaturePoints, cv::noArray());
                    cv::sfm::essentialFromFundamental(f, camMatrix, camMatrix, e);
                    // cv::findEssentialMat(prevFeaturePoints, curFeaturePoints, focal[0], pp);

                    std::vector<cv::Mat> rvec;
                    std::vector<cv::Mat> tvec;

                    cv::sfm::motionFromEssential(e, rvec, tvec);

                    cv::Mat pt1 = (Mat_<double>(2,1) << prevFeaturePoints.front().x, prevFeaturePoints.front().y);
                    cv::Mat pt2 = (Mat_<double>(2,1) << curFeaturePoints.front().x, curFeaturePoints.front().y);

                    int id = cv::sfm::motionFromEssentialChooseSolution(rvec, tvec, camMatrix, pt1, camMatrix, pt2);

                    R = rvec[id];
                    t = tvec[id];

                    cv::Mat p1, p2;

                    cv::sfm::projectionFromKRt(camMatrix, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), p1);
                    cv::sfm::projectionFromKRt(camMatrix, R, t, p2);

                    tTotal = tTotal + rTotal * t;
                    rTotal = rTotal * R;

                    std::vector<cv::Mat> Ps;
                    Ps.push_back(p1);
                    Ps.push_back(p2);

                    std::vector<Mat_<double> > pts(2);
                    pts[0].create(2, prevFeaturePoints.size());
                    pts[1].create(2, curFeaturePoints.size());
                    for (auto i = 0; i < prevFeaturePoints.size(); ++i) {
                        pts[0].row(0).col(i) = prevFeaturePoints[i].x;
                        pts[0].row(1).col(i) = prevFeaturePoints[i].y;
                    }
                    for (auto i = 0; i < curFeaturePoints.size(); ++i) {
                        pts[1].row(0).col(i) = curFeaturePoints[i].x;
                        pts[1].row(1).col(i) = curFeaturePoints[i].y;
                    }

                    cv::Mat points3d;

                    cv::sfm::triangulatePoints(pts, Ps, points3d);

                    for (auto i = 0; i <points3d.cols; ++i) {
                        auto x = points3d.at<double>(0, i);
                        auto y = points3d.at<double>(1, i);
                        auto z = -points3d.at<double>(2, i);

                        if (fabs(x) < 1e2 && fabs(y) < 1e2 && fabs(z) < 1e2) {
                            cv::Mat pointVector = (Mat_<double>(3,1) << x, y, z);
                            pointVector = rTotal * pointVector + tTotal;
                            cv::Point3f point(pointVector.at<double>(0),
                                              pointVector.at<double>(1),
                                              pointVector.at<double>(2));
                            points.push_back(point);
                            colors.push_back(prevFrame.at<cv::Vec4b>(prevFeaturePoints[i].x, prevFeaturePoints[i].y));

                        }

                    }

                    cv::Mat renderFrame;
                    curFrame.copyTo(renderFrame);
                    drawPoints(renderFrame, curFeaturePoints);
                    drawLines(renderFrame, prevFeaturePoints, curFeaturePoints);
                    cv::imshow("Features", renderFrame);
                    curFrame.copyTo(prevFrame);
                    prevFeaturePoints = curFeaturePoints;

                    cv::Affine3d cameraPose(rTotal, tTotal);
                    trajectoryPoints.push_back(cameraPose);

                    cv::viz::WTrajectory trajectory(trajectoryPoints);
                    vizWindow.showWidget("trajectory", trajectory);

                    cv::viz::WTrajectoryFrustums trajectoryFrustums(trajectoryPoints, focal, 0.002);
                    vizWindow.showWidget("trajectoryFrustums", trajectoryFrustums);

                    if (!points.empty()) {
                        cv::viz::WCloud cloud(points, colors);
                        cloud.setRenderingProperty(cv::viz::POINT_SIZE, 2);
                        vizWindow.showWidget("Point cloud", cloud);
                    }

                   // vizWindow.

               }

                vizWindow.spinOnce(15,  true);
                cv::waitKey(150);
                ++i;
                // if (i > 60) break;
            }

        } while (!captureFrame.empty());

        vizWindow.spin();

    }

    return 0;
}

