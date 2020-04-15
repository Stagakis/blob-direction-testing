#include <img_proc_func.h>
#include <helpers.h>
#include <chrono> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ThreeFrameProcesser{


    public:
        cv::Mat current, previous, preprevious;
        cv::Mat diff_img, diff_cur_prev, diff_prev_preprev;
        cv::Mat visible_parts_cur, visible_parts_prev;
        
        ThreeFrameProcesser(cv::Mat cur, cv::Mat prev, cv::Mat preprev):current(cur),previous(prev),preprevious(preprev){}

        void calculateDifferences(int threshold){
            auto start = std::chrono::high_resolution_clock::now();
            two_image_differencing(current, previous, diff_cur_prev, threshold, 190); //Diff between current(t) and prev(t-1)
            two_image_differencing(previous, preprevious, diff_prev_preprev, threshold, 105); //Diff between prev(t-1) and pre_prev(t-2)
            add(diff_cur_prev, diff_prev_preprev, diff_img); //NOTE: color saturates to 255, does not overflow
            cout << "Diff_img calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << endl;
        }

        void calculateVisibleParts(cv::Mat outImg){
            Mat visible_parts_cur;
            cv::bitwise_and(current, current, visible_parts_cur, diff_cur_prev);
            Mat visible_parts_prev;
            cv::bitwise_and(previous, previous, visible_parts_prev, diff_prev_preprev);

            hconcat(visible_parts_cur, visible_parts_prev, outImg);

        }

        void two_image_differencing(const cv::Mat &img1, const cv::Mat &img2, cv::Mat outImage, int threshold, int color){
            outImage = img1 - img2;
            cv::cvtColor(img1, outImage, cv::COLOR_BGR2GRAY);
            cv::threshold(img1, outImage, threshold, color, cv::THRESH_BINARY);
        }
};

