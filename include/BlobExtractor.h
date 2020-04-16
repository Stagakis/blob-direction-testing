#pragma once

#include <opencv2/core/core.hpp>

#define PIXEL_THRESHOLD 30

class BlobExtractor{

    public:
        cv::Mat diff_img;
        cv::Mat blob_img;
        std::vector<cv::Rect> blob_rects;
        std::vector<cv::Mat> blob_img_mask; //Binary image with the same size as blob_img
        int num_of_blobs;


        BlobExtractor(cv::Mat diff_img);
        void ExtractBlobs();
        void recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img, uchar mcolor = 255);

        void GetBlob(int index, cv::Mat& outImage);
        cv::Mat& GetBlobDilated(int index, int dilation_kernel_size);
        cv::Mat& GetRectOfBlob(int index);
        bool isValid(cv::Mat& blob);
};