#include <iostream>
#include <helpers.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d.hpp>
using namespace std;
using namespace cv;

#define GET_VARIABLE_NAME(Variable) (#Variable)
#define CHECK_IMAGE(mat_name, wait) cv::namedWindow(GET_VARIABLE_NAME(mat_name), cv::WINDOW_KEEPRATIO); cv::imshow(GET_VARIABLE_NAME(mat_name), mat_name); if(wait) cv::waitKey(0);

void my_matching_method(Mat& queryDescriptors, Mat& trainDescriptors, std::vector<DMatch>& matches, InputArray mask){
    matches.clear();
    int best_distance;
    for(int i = 0; i< queryDescriptors.rows; i++){
        cv::Mat qDesc = queryDescriptors.row(i);
        int best_dist = 1000000000;
        int best_index = 0;
        for(int j = 0; j< trainDescriptors.rows; j++){
            cv::Mat tDesc = trainDescriptors.row(j);
            int dist = DescriptorDistance(qDesc, tDesc);
            if (dist<best_dist){
                best_index=j;
                best_dist = dist;
            }
        }

        matches.push_back(DMatch(i, best_index, best_dist));
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


int pixel_threshold = 30;

void update_hsv_image(cv::Mat& hsv_img, float angle, const cv::Mat& mask_img){

    Mat temp(Size(hsv_img.cols, hsv_img.rows), CV_8UC3, cv::Scalar(angle/2, 255, 255));
    Mat template_img_3d;
    cv::cvtColor(mask_img, template_img_3d, cv::COLOR_GRAY2RGB);

    bitwise_and(template_img_3d, temp, temp);
    add(temp, hsv_img, hsv_img);
}

Vec2f calculate_direction_com(cv::Mat& image) {
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

void filter_keypoints_and_descriptors(vector<KeyPoint>& all_kp, Mat& all_des, vector<KeyPoint>& out_kp, Mat& out_des, int color, const cv::Mat& mask_img){
    for(int i = 0; i< all_kp.size(); i++){
        Point2d test_point = all_kp[i].pt;
        if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp.push_back(all_kp[i]);
            out_des.push_back(all_des.row(i));
        }
    }
}

void filter_keypoints(vector<KeyPoint>& all_kp, vector<KeyPoint>& out_kp, int color, const cv::Mat& mask_img){
    for(int i = 0; i< all_kp.size(); i++){
        Point2d test_point = all_kp[i].pt;
        if(mask_img.at<uchar>(test_point) == color || mask_img.at<uchar>(test_point) == 255){
            out_kp.push_back(all_kp[i]);
        }
    }
}

void create_mask_mat(cv::Mat& mask_mat, vector<KeyPoint>& kp_cur_blob, vector<KeyPoint>& kp_prev_blob, int angle, int tolerance){
    mask_mat = cv::Mat::zeros(Size(kp_prev_blob.size(), kp_cur_blob.size()), CV_8UC1);
    for(int i = 0; i < mask_mat.rows; i++){
        for (int j = 0 ; j< mask_mat.cols; j++){
            Point from = kp_prev_blob[i].pt;
            Point to = kp_cur_blob[j].pt;
            Vec2f keypoint_dir = Vec2f(to.y - from.y, to.x - from.x);
            int keypont_angle = floor(atan2(-keypoint_dir.val[0], keypoint_dir.val[1]) * 180 / 3.14159265);
            if (keypont_angle < 0) keypont_angle += 360;

            if( abs(keypont_angle - angle) < tolerance )
            {
                mask_mat.at<uchar>(i,j) = 1;
                //cout << "Keypont(angle) "<< " is " << keypont_angle <<endl;
            }
        }
    }
    cout<< "kp_prev_blob Size: " << kp_prev_blob.size() << endl;
    cout<< "kp_cur_blob Size: " << kp_cur_blob.size() << endl;
    cout<< "MASK Size: " << mask_mat.size() << endl;
    mask_mat = mask_mat.t();
}

int calculate_angle_by_com(cv::Mat& blob_image_bb){

    Vec2f dir = calculate_direction_com(blob_image_bb);
    int angle = floor(atan2(-dir.val[0], dir.val[1]) * 180 / 3.14159265);
    if (angle < 0) angle += 360;
    //cout << "Blob angle is: " << angle << endl;
    return angle;
}

void calc_optical_flow_gt(cv::Mat& frame1, cv::Mat& frame2, cv::Mat& out){
    Mat prvs;
    Mat next;
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    cvtColor(frame2, next, COLOR_BGR2GRAY);

    cvtColor(frame2, next, COLOR_BGR2GRAY);
    Mat flow(prvs.size(), CV_32FC2);
    calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    // visualization
    Mat flow_parts[2];
    split(flow, flow_parts);
    Mat magnitude, angle, magn_norm;
    cartToPolar(flow_parts[1], -flow_parts[0], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    Mat _hsv[3], hsv, hsv8, optical_flow_gt;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, out, COLOR_HSV2BGR);

}