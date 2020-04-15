#pragma once

#include <opencv2/core/core.hpp>

class ThreeFrameProcesser{


    public:
        cv::Mat current, previous, preprevious;
        cv::Mat diff_img, diff_cur_prev, diff_prev_preprev;
        cv::Mat visible_parts_cur, visible_parts_prev;

        
        ThreeFrameProcesser(cv::Mat cur, cv::Mat prev, cv::Mat preprev);
        void calculateDifferences(int threshold);
        void calculateVisibleParts(cv::Mat outImg);
        void two_image_differencing(const cv::Mat &img1, const cv::Mat &img2, cv::Mat outImage, int threshold, int color);
};

