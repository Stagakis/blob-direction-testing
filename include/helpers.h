#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
extern int frame_counter;

#define GET_VARIABLE_NAME(Variable) (#Variable)
#define CHECK_IMAGE(mat_name, wait) cv::namedWindow(GET_VARIABLE_NAME(mat_name), cv::WINDOW_KEEPRATIO); cv::imshow(GET_VARIABLE_NAME(mat_name), mat_name); if(wait) cv::waitKey(0);
//int frame_counter;
// Function for calculating median 
void my_matching_method(Mat& queryDescriptors, Mat& trainDescriptors, std::vector<DMatch>& matches, InputArray mask=noArray() );
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
double findMedian(vector<float> a);
void recursion_func(Point pixel, Mat& img, uchar blob_number, Mat& template_img, uchar mcolor = 255);
Vec2f calculate_direction(cv::Mat& image);
Vec2f calculate_direction_com(const cv::Mat& image);
void update_hsv_image(cv::Mat& hsv_img, float angle, const cv::Mat& mask_img);
void filter_keypoints_indeces(const vector<cv::KeyPoint>& all_kp, vector<size_t>& out_kp_indeces, int color,
                         cv::Mat& mask_img, cv::Rect& bb, int range = 1);
void filter_keypoints_and_descriptors(vector<KeyPoint>& all_kp, Mat& all_des, vector<KeyPoint>& out_kp, Mat& out_des, int color, const cv::Mat& mask_img);
void filter_keypoints(vector<KeyPoint>& all_kp, vector<KeyPoint>& out_kp, int color, const cv::Mat& mask_img);
void create_mask_mat(cv::Mat& mask_mat, vector<KeyPoint>& kp_cur_blob, vector<KeyPoint>& kp_prev_blob, int angle, int tolerance);

int calculate_angle_by_com(const cv::Mat& blob_image_bb);

void calc_optical_flow_gt(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& out);
//