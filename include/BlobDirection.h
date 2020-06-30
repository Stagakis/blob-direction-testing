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

#include <ThreeFrameProcesser.h>
#include <BlobExtractor.h>

using namespace cv;
using namespace std;

int best_dist_threshold = 100;

#define LOG(x) std::cout << x << endl;
#define PRINT_VECTOR(x) for(auto k: x) std::cout<< std::setprecision(4) << k<< ", ";
#define PRINT_VECTOR_SBS(x, y) for(int i = 0; i <x.size(); i++) std::cout<<x[i]<< "  " << y[i] << std::endl;
#define MEAN_VALUE(v) std::accumulate(v.begin(), v.end(), 0LL) / v.size()
#define PRINT_KEYPOINT(x) for(auto k: x) std::cout<<k.response<<endl;

#define GET_VARIABLE_NAME(Variable) (#Variable)
#define CHECK_IMAGE(mat_name, wait) cv::namedWindow(GET_VARIABLE_NAME(mat_name), cv::WINDOW_KEEPRATIO); cv::imshow(GET_VARIABLE_NAME(mat_name), mat_name); if(wait) cv::waitKey(0);

#define PI 3.14159265


void createWindowsAndTrackbars();
static void frame_trackbar(int, void*);
void categorize_kp_to_blobs();

//Mat diff_img;
Mat current_greyscale, previous_greyscale;

ORB_SLAM2::ORBextractor* ORBextractorLeft;
BlobExtractor* blextr;
ThreeFrameProcesser* tfp;

std::vector<cv::Mat> images;
int frame_slider = 387; // 223;
int threshold_slider = 40;// 52; //40;
int keypoint_filter_range_slider = 1;// 52; //40;
int angle_tolerance_slider = 50;
int adapt_neighboorhood = 10;// 52; //40;
int adapt_constant = 10;// 52; //40;
int use_basic_thresholding = 1;
int blobs_to_process_slider = 4;

int dilation_slider = 1; //40;
int recursion_depth = 0;
int template_value = 0;
int match_result_value = 0;


//Timing variables
vector<long long> blob_extractions_time;
vector<long long> calc_dir_time;
vector<long long> dilations_time;


cv::Mat diff_image; //This is only for lamda usage

//Keypoints and descriptros of ORB
vector<KeyPoint> kp_cur, kp_prev;
Mat des_cur, des_prev;

std::vector<cv::Mat> templates;
std::vector<cv::Mat> list_of_match_results;
std::vector<cv::Mat> list_of_match_results_withcrosscheck;
//std::vector<Vec2f> directions;
std::vector<int> angle_per_blob;

int brute_force_whole_img_num_of_matches;
int mask_num_of_matches;
int crosscheck_num_of_matches;


//TIMING VARIABLES
unsigned long long time_extracing_orb_features;
unsigned long long time_blob_extraction;
unsigned long long time_three_frame_differencing;
unsigned long long time_dilation, time_dilation_total;
unsigned long long time_keypoint_filtering, time_keypoint_filtering_total;
unsigned long long time_angle_calculation, time_angle_calculation_total;
unsigned long long time_matching_without_mask, time_matching_without_mask_total;
unsigned long long time_matching_with_mask, time_matching_with_mask_total;
unsigned long long time_orb_brute_force_whole_image;