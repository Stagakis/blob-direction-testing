#pragma once

#include <opencv2/core/core.hpp>

#define PIXEL_THRESHOLD 30

class BlobExtractor{

    public:
        cv::Mat diff_img;
        cv::Mat blob_img;
        std::vector<cv::Point> white_pixels;
        std::vector<cv::Rect> blob_rects;
        int num_of_blobs;


        BlobExtractor(cv::Mat diff_img);
        void ExtractBlobs();
        void recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img, uchar mcolor = 255);

        cv::Mat& GetBlob(int index);
        cv::Mat& GetBlobDilated(int index, int dilation_kernel_size);
        cv::Mat& GetRectOfBlob(int index);
        bool isValid(cv::Mat& blob);
};