#pragma once

#include <opencv2/core/core.hpp>

#define PIXEL_THRESHOLD 30

class BlobExtractor{

    public:
        cv::Mat diff_img;
        cv::Mat diff_img_cur_prev, diff_img_prev_preprev;
        cv::Mat blob_img;
        std::vector<cv::Rect> blob_rects;
        std::vector<cv::Mat> blob_img_mask; //Binary image with the same size as blob_img
        int num_of_blobs;


        //BlobExtractor(cv::Mat _diff_img);
        BlobExtractor(cv::Mat _diff_img, cv::Mat _diff_img_cur_prev, cv::Mat _diff_img_prev_preprev);
        void ExtractBlobs();
        void recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img, uchar mcolor = 255);

        void GetBlob(int index, cv::Mat& outImage);
        void GetBlobDilated(int index, cv::Mat& outImage, int dilation_kernel_size=3);
        cv::Mat& GetRectOfBlob(int index);
        bool isValid(cv::Mat& blob);
};