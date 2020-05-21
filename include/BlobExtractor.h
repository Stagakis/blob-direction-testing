#pragma once

#include <opencv2/core/core.hpp>

#define PIXEL_THRESHOLD 20
#define THRESHOLD_VALUE_PREV 95
#define THRESHOLD_VALUE_CUR 160

class BlobExtractor{

    public:
        cv::Mat diff_img;
        //cv::Mat diff_img_cur_prev, diff_img_prev_preprev;
        //std::vector<cv::Rect> blob_rects;
        //std::vector<cv::Mat> blob_img_mask; //Binary image with the same size as blob_img
        std::vector<cv::Mat> blob_img_full; //Non-Binary image with the same size as blob_img
        std::vector<cv::Rect> blob_rects;
        std::vector<cv::Point> white_pixels;
        int start;
        int num_of_blobs;
        int scale_factor = 1;
        double blob_extraction_total_time;

        //BlobExtractor(cv::Mat _diff_img);
        //BlobExtractor(const cv::Mat& _diff_img, const cv::Mat& _diff_img_cur_prev, const cv::Mat& _diff_img_prev_preprev);
        BlobExtractor();
        BlobExtractor(cv::Mat& _diff_img);
        bool GetNextBlob(cv::Mat& out_blob_img, cv::Rect& out_bb);
        void GetBlobFullSize(int index, cv::Mat& outImage);
        //void ExtractBlobs();
        void ExtractBlobs(cv::Mat& diff_img);
        //void recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img, uchar mcolor = 255);
        void recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img,
                int& before_points, int& after_points,
                uchar mcolor = 255);
        void recursion_func2(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img,
                        int& before_points, int& after_points,
                        cv::Point &top_left, cv::Point& bottom_right, uchar mcolor = 255);
    //void Downscale();
        //void GetBlobDilated(int index, cv::Mat& outImage, int dilation_kernel_size=3);
        //cv::Mat& GetRectOfBlob(int index);
        bool isValid(const cv::Mat& blob);
};
