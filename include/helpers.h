#pragma once

#include <stdio.h>
#include <vector>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;



// Function for calculating median 
double findMedian(vector<float> a);
void recursion_func(Point pixel, Mat& img, uchar blob_number, Mat& template_img, uchar mcolor = 255);
Vec2f calculate_direction(cv::Mat& image);

//