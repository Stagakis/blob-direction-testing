#pragma once

#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>
#include <ORBextractor.h>

using namespace cv;
using namespace std;


#define LOG(x) std::cout << x << endl;
#define PRINT_VECTOR(x) for(auto k: x) std::cout<<k<<endl;
#define PRINT_VECTOR_SBS(x, y) for(int i = 0; i <x.size(); i++) std::cout<<x[i]<< "  " << y[i] << std::endl;
#define MEAN_VALUE(v) std::accumulate(v.begin(), v.end(), 0LL) / v.size()
#define PRINT_KEYPOINT(x) for(auto k: x) std::cout<<k.response<<endl;
#define CHECK_IMAGE(name, x, wait) cv::namedWindow(name, cv::WINDOW_KEEPRATIO); cv::imshow(name, x); if(wait) cv::waitKey(0);
#define PI 3.14159265


void createWindowsAndTrackbars();
static void frame_trackbar(int, void*);
void categorize_kp_to_blobs();

Mat diff_img;
Mat current_greyscale, previous_greyscale;


ORB_SLAM2::ORBextractor* ORBextractorLeft;
std::vector<cv::Mat> images;
std::vector<Point> white_pixels;
int frame_slider = 30; // 223;
int threshold_slider = 40;
int recursion_depth = 0;
int template_value = 0;

//Timing variables
vector<long long> blob_extractions_time;
vector<long long> calc_dir_time;
vector<long long> dilations_time;




//Keypoints and descriptros of ORB
vector<KeyPoint> kp_cur, kp_prev;
Mat des_cur, des_prev;

std::vector<cv::Mat> templates;
std::vector<Vec2f> directions;
std::vector<float> angle_per_blob;

std::vector<cv::Mat> blobs;
