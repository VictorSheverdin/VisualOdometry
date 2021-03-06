#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
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

void featureTracking(Mat sourceFrame, Mat targetFrame, vector<Point2f>& points1, vector<Point2f>& points2 )
{
    cv::Mat grey1;
    cv::cvtColor(sourceFrame, grey1, COLOR_BGR2GRAY);

    cv::Mat grey2;
    cv::cvtColor(targetFrame, grey2, COLOR_BGR2GRAY);

    vector<Point2f> trackingPoints;
    vector<Point2f> resultPoints;

    /*static auto detector = cv::xfeatures2d::SIFT::create( 5000 );

    std::vector< cv::KeyPoint > sourceKeypoints;
    cv::Mat sourceDescriptors;
    detector->detectAndCompute( grey1, cv::noArray(), sourceKeypoints, sourceDescriptors );

    std::vector< cv::KeyPoint > targetKeypoints;
    cv::Mat targetDescriptors;
    detector->detectAndCompute( grey2, cv::noArray(), targetKeypoints, targetDescriptors );

    static auto  matcher = cv::BFMatcher::create();

    std::vector< DMatch > matches;
    matcher->match( sourceDescriptors, targetDescriptors, matches );

    for (auto i = 0; i < sourceKeypoints.size(); ++i ) {
        trackingPoints.push_back( sourceKeypoints[ i ].pt );

    }*/

    cv::goodFeaturesToTrack( grey1, trackingPoints, 10000, 0.01, 10, cv::noArray(), 3, false, 0.04);

    vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK( grey1, grey2, trackingPoints, resultPoints, status, err );

    points1.clear();
    points2.clear();

    if ( !status.empty() && !err.empty() ) {

        float minErr = err.front();
        float maxErr = err.front();

        for ( auto i = 1; i < err.size(); ++i ) {
            minErr = min( minErr, err[i] );
            maxErr = max( maxErr, err[i] );
        }

        for (auto i = 0; i < status.size(); ++i ) {
            if ( status[i] && err[i] < minErr + 0.05 * ( maxErr - minErr ) ) {
                points1.push_back( trackingPoints[i] );
                points2.push_back( resultPoints[i] );
            }

        }

    }

}

void drawPoints(Mat frame, vector<Point2f>& points, const cv::Scalar &color = cv::Scalar(0, 0, 255, 255) )
{
    for (auto &i : points)
        cv::circle(frame, i, 5, color, -1);

}

void drawLines(Mat frame, vector<Point2f>& points1, vector<Point2f>& points2)
{
    for (int i = 0; i <points1.size(); ++i)
        cv::line(frame, points1[i], points2[i], cv::Scalar(0, 255, 0, 255));

}

int main( int argc, char** argv )
{
    namedWindow( "Features", WINDOW_KEEPRATIO );
    cv::resizeWindow("Features", 800, 600);
    cv::moveWindow("Features", 80, 10);

    cv:viz::Viz3d vizWindow("Viz");
    cv::moveWindow("Viz", 500, 10);
    vizWindow.showWidget("coordSystemWidget", cv::viz::WCoordinateSystem());


    Mat camMatrix = (Mat_<double>(3,3) << 9.1073236417442979e+02, 0., 1.0011878386845302e+03, 0.,
                     9.0913115226566401e+02, 1.0080606237570406e+03, 0., 0., 1.);

    std::vector<double> distCoefficients = { 2.5205656544025762e-02, -1.9577404877149228e-02,
                                             -8.4204471895308317e-04, -2.6845992072923154e-03, 0. };

    cv::Vec2d focal(9.1073236417442979e+02, 9.0913115226566401e+02);
    cv::Point2d pp(1.0011878386845302e+03, 1.0080606237570406e+03);

    std::vector<cv::Affine3d> trajectoryPoints;

    cv::Mat captureFrame;
    cv::Mat prevFrame;
    vector<cv::Point2f> prevFeaturePoints;
    cv::Mat curFrame;
    vector<cv::Point2f> curFeaturePoints;

    int i = 10000;

    cv::Mat rTotal = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat tTotal = cv::Mat::zeros(3, 1, CV_64F);

    std::vector<cv::Scalar> tempColors;

    std::vector<cv::Point3f> points;
    std::vector<cv::Vec4b> colors;

    do {

        char fileName[256];
        sprintf(fileName, "/home/victor/datasets/Polygon/left/%05d_left.jpg", i);
        captureFrame = imread(fileName);

        if (!captureFrame.empty()) {

            cv::Mat frame;

            cv::undistort(captureFrame, frame, camMatrix, distCoefficients);

            if ( !prevFrame.empty() ) {
                frame.copyTo(curFrame);
                featureTracking(prevFrame, curFrame, prevFeaturePoints, curFeaturePoints );

                cv::Mat renderFrame;
                curFrame.copyTo(renderFrame);
                drawPoints( renderFrame, curFeaturePoints );
                drawLines(renderFrame, prevFeaturePoints, curFeaturePoints);

                cv::imshow("Features", renderFrame);

                if ( prevFeaturePoints.size() > 8 ) {

                    Mat R, t;
                    cv:: Mat f;
                    cv:: Mat e;
                    f = cv::findFundamentalMat( prevFeaturePoints, curFeaturePoints, cv::noArray() );
                    e = camMatrix.t() * f * camMatrix;

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

                    std::vector<cv::Mat> rvec;
                    std::vector<cv::Mat> tvec;

                    cv::sfm::motionFromEssential(e, rvec, tvec);

                    cv::Mat pt1 = (Mat_<double>(2,1) << prevFeaturePoints.front().x, prevFeaturePoints.front().y);
                    cv::Mat pt2 = (Mat_<double>(2,1) << curFeaturePoints.front().x, curFeaturePoints.front().y);

                    int id = cv::sfm::motionFromEssentialChooseSolution(rvec, tvec, camMatrix, pt1, camMatrix, pt2);

                    R = rvec[id];
                    t = tvec[id];

                    // cv::recoverPose( e, prevFeaturePoints, curFeaturePoints, R, t );

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


                    points.clear();
                    colors.clear();

                    for (auto i = 0; i <points3d.cols; ++i) {
                        auto x = -points3d.at<double>(0, i);
                        auto y = -points3d.at<double>(1, i);
                        auto z = -points3d.at<double>(2, i);

                        if (fabs(x) < 1e2 && fabs(y) < 1e2 && fabs(z) < 1e2) {
                            cv::Mat pointVector = (Mat_<double>(3,1) << x, y, z);
                            pointVector = rTotal * pointVector + tTotal;
                            cv::Point3f point(pointVector.at<double>(0),
                                              pointVector.at<double>(1),
                                              pointVector.at<double>(2));
                            points.push_back(point);
                            colors.push_back( curFrame.at< cv::Vec4b >( curFeaturePoints[i].x, curFeaturePoints[i].y ) );

                        }

                    }

                    cv::Affine3d cameraPose(rTotal, tTotal);
                    trajectoryPoints.push_back(cameraPose);

                    cv::viz::WTrajectory trajectory(trajectoryPoints);
                    vizWindow.showWidget("trajectory", trajectory);

                    cv::viz::WTrajectoryFrustums trajectoryFrustums(trajectoryPoints, focal, 0.002);
                    vizWindow.showWidget("trajectoryFrustums", trajectoryFrustums);

                    if (!points.empty()) {
                        cv::viz::WCloud cloud( points, colors );
                        cloud.setRenderingProperty(cv::viz::POINT_SIZE, 2);
                        cloud.setColor(cv::viz::Color(0, 255, 255));
                        vizWindow.showWidget("Point cloud", cloud);
                    }

                }
                else
                    std::cout << "empty" << std::endl;

           }

            frame.copyTo(prevFrame);

            vizWindow.spinOnce(15,  true);
            cv::waitKey(15);
            ++i;

        }

    } while (!captureFrame.empty());

    vizWindow.spin();


    return 0;
}

