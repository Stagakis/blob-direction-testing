#include <iostream>
#include <helpers.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


int pixel_threshold = 30;

void update_hsv_image(cv::Mat& hsv_img, float angle, const cv::Mat& mask_img){

    Mat temp(Size(hsv_img.cols, hsv_img.rows), CV_8UC3, cv::Scalar(angle/2, 255, 255));
    Mat template_img_3d;
    cv::cvtColor(mask_img, template_img_3d, cv::COLOR_GRAY2RGB);

    bitwise_and(template_img_3d, temp, temp);
    add(temp, hsv_img, hsv_img);
}

Vec2f calculate_direction2(cv::Mat& image) {
    cv::Vec2i start_point(0, 0);
    cv::Vec2i end_point(0, 0);
    int before_points = 0;
    int after_points = 0;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) == 105) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                before_points++;
            }

            if (image.at<uchar>(i, j) == 190) {
                end_point.val[0] += i;
                end_point.val[1] += j;
                after_points++;
            }

            if (image.at<uchar>(i, j) == 255) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                end_point.val[0] += i;
                end_point.val[1] += j;
            }
        }
    }

    end_point /= after_points;
    start_point /= before_points;

    cv::Vec2f direction = end_point - start_point;
    float length = sqrt(direction.dot(direction));

    return direction / length;

}


Vec2f calculate_direction(cv::Mat& image) {
    cv::Vec2i start_point(0, 0);
    cv::Vec2i end_point(0, 0);
    int before_points = 0;
    int after_points = 0;
    //int window_size = image.cols * image.rows;
    //cout << "Entered Direction " << endl;    

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) == 105) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                before_points++;
            }

            if (image.at<uchar>(i, j) == 190) {
                end_point.val[0] += i;
                end_point.val[1] += j;
                after_points++;
            }

            if (image.at<uchar>(i, j) == 255) {
                start_point.val[0] += i;
                start_point.val[1] += j;
                end_point.val[0] += i;
                end_point.val[1] += j;
            }
        }
    }
    if (before_points == 0 || after_points == 0) {
        return Vec2f(2, 2);
    }

    end_point /= after_points;
    start_point /= before_points;

    cv::Vec2f direction = end_point - start_point;
    float length = sqrt(direction.dot(direction));

    //std::cout << "Total: " << before_points + after_points << endl;
    //check_image("Blob", image);

    if (length == 0 || before_points + after_points < pixel_threshold || before_points < after_points / 2 || after_points < before_points / 2) {

        //std::cout << "calcdir test " << before_points << " " << after_points << " " << length << " " << endl;
        return Vec2f(2, 2);
    }
    else {

        return direction / length;
    }
}

double findMedian(vector<float> a)
{
    int n = a.size();
    // First we sort the array 
    sort(a.begin(), a.end());

    // check for even case 
    if (n % 2 != 0)
        return (double)a[n / 2];

    return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
}

void recursion_func(Point pixel, Mat& img, uchar blob_number, Mat& template_img, uchar mcolor) {
    //recursion_depth++;
    //cout << "Recursion is called with imgSize " << img.size << " template_imgSize " << template_img.size 
    //    << " And pixel " << pixel.val[0] << "," << pixel.val[1]
    //    << " recursion depth " << recursion_depth << endl;

    if (img.at<uchar>(pixel) == 0)
        return;

    int current_color = img.at<uchar>(pixel);

    if (current_color != 255 && mcolor != 255 && mcolor != current_color)
        return;

    img.at<uchar>(pixel) = 0;
    template_img.at<uchar>(pixel.y - template_img.rows, pixel.x - template_img.cols) = blob_number;

    recursion_func(Point(pixel.x + 0, pixel.y + 1), img, blob_number, template_img, current_color);

    recursion_func(Point(pixel.x + 1, pixel.y + 1), img, blob_number, template_img, current_color);
    recursion_func(Point(pixel.x - 1, pixel.y + 1), img, blob_number, template_img, current_color);

    recursion_func(Point(pixel.x + 0, pixel.y - 1), img, blob_number, template_img, current_color);

    recursion_func(Point(pixel.x + 1, pixel.y - 1), img, blob_number, template_img, current_color);

    recursion_func(Point(pixel.x - 1, pixel.y - 1), img, blob_number, template_img, current_color);


    recursion_func(Vec2i(pixel.x + 1, pixel.y + 0), img, blob_number, template_img, current_color);
    recursion_func(Vec2i(pixel.x - 1, pixel.y + 0), img, blob_number, template_img, current_color);

}