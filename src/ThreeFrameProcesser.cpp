#include <ThreeFrameProcesser.h>
#include <helpers.h>
#include <chrono> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


ThreeFrameProcesser::ThreeFrameProcesser(cv::Mat &cur, cv::Mat &prev, cv::Mat &preprev):current(cur),previous(prev),preprevious(preprev){}

void ThreeFrameProcesser::calculateDifferences(int threshold){
    auto start = std::chrono::high_resolution_clock::now();
    two_image_differencing(current, previous, diff_cur_prev, threshold, 190); //Diff between current(t) and prev(t-1)
    two_image_differencing(previous, preprevious, diff_prev_preprev, threshold, 105); //Diff between prev(t-1) and pre_prev(t-2)
    add(diff_cur_prev, diff_prev_preprev, diff_img); //NOTE: color saturates to 255, does not overflow
    cout << "Diff_img calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << endl;
}

void ThreeFrameProcesser::calculateVisibleParts(cv::Mat &outImg){
    Mat visible_parts_cur;
    cv::bitwise_and(current, current, visible_parts_cur, diff_cur_prev);
    Mat visible_parts_prev;
    cv::bitwise_and(previous, previous, visible_parts_prev, diff_prev_preprev);

    hconcat(visible_parts_cur, visible_parts_prev, outImg);
}

void ThreeFrameProcesser::two_image_differencing(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &outImage, int threshold, int color){
    outImage = img1 - img2;
    cv::cvtColor(outImage, outImage, cv::COLOR_BGR2GRAY);
    cv::threshold(outImage, outImage, threshold, color, cv::THRESH_BINARY);
}


