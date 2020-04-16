#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>
#include <ORBextractor.h>
#include <BlobDirection.h>
#include <helpers.h>
#include <chrono> 
#include <numeric> 

#include <ThreeFrameProcesser.h>
#include <BlobExtractor.h>

using namespace cv;
using namespace std;
using namespace std::chrono;


static void threshold_trackbar (int , void* )
{
    cv::Mat current, previous, pre_previous;
    current = images[frame_slider];
    previous = images[frame_slider - 1];
    pre_previous = images[frame_slider - 2];

    ThreeFrameProcesser tfp(current, previous, pre_previous);
    tfp.calculateDifferences(threshold_slider);
    cv::Mat visible_parts;
    tfp.calculateVisibleParts(visible_parts);

    auto start = high_resolution_clock::now();
    cv::cvtColor(current, current_greyscale, cv::COLOR_BGR2GRAY);
    ORBextractorLeft->operator()(current_greyscale, NULL, kp_cur, des_cur);

    cv::cvtColor(previous, previous_greyscale, cv::COLOR_BGR2GRAY);
    ORBextractorLeft->operator()(previous_greyscale, NULL, kp_prev, des_prev);
    cout << "ORB Extraction time*2: " << duration_cast<microseconds>(high_resolution_clock::now() - start).count() << endl;

    {
        auto start = high_resolution_clock::now();
        auto bf = BFMatcher(cv::NORM_HAMMING, true);
        std::vector< DMatch > matches;
        bf.match(des_prev, des_cur, matches);
        Mat match_results;
        drawMatches(previous, kp_prev, current, kp_cur, matches, match_results, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cout << "Calculating all matches from brute force: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        CHECK_IMAGE("WholeImageBruteForce", match_results, false);
    }
    LOG("TEST");
    LOG(tfp.diff_img.size());
    imshow("Diff", tfp.diff_img);
    LOG("TEST2");
    namedWindow("VisiblePart1", WINDOW_FREERATIO);
    imshow("VisiblePart1", visible_parts);

////
    start = high_resolution_clock::now();
    Mat diff_img_enlarged = Mat::zeros(Size(3*tfp.diff_img.cols, 3*tfp.diff_img.rows), CV_8U);
    Rect diff_img_clone_center = Rect(tfp.diff_img.cols, tfp.diff_img.rows, tfp.diff_img.cols,tfp.diff_img.rows);
    tfp.diff_img.copyTo(diff_img_enlarged(diff_img_clone_center));

    Mat diff_img_white_only;
    cv::threshold(diff_img_enlarged, diff_img_white_only, 254, 255, cv::THRESH_BINARY);
    cv::findNonZero(diff_img_white_only, white_pixels);
    cout << "White pixel extraction: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;



    Mat hsv_image = Mat::zeros(Size(tfp.diff_img.cols, tfp.diff_img.rows), CV_8UC3);   
    BlobExtractor blextr(tfp.diff_img);
    blextr.ExtractBlobs();
    for (int i = 0; i < blextr.num_of_blobs; i++){
        vector<KeyPoint> kp_cur, kp_prev;
        vector<Mat> des_per_template_cur(blextr.num_of_blobs), des_per_template_prev(blextr.num_of_blobs);
        Vec2f dir = calculate_direction2(blextr.GetBlob(i));

        int angle = floor(atan2(-dir.val[0], dir.val[1]) * 180 / PI);
        if (angle < 0) angle += 360;
        cout << "Direction of Blob No. " << i << " is " << angle;


    }

    //BlobMatcher blmatch; Takes the Rects
////

    templates.clear();
    directions.clear();
    angle_per_blob.clear();

    calc_dir_time.clear();
    blob_extractions_time.clear();
    dilations_time.clear();

    int custom_k = 0;
    int template_number = 0;
    std::cout << "==============" << endl;
    std::vector<float> dir_x, dir_y;





    for (int k = 0; k < white_pixels.size(); k++) {
        
        Point pixel = white_pixels[k]; // Point(white_pixels[k].y, white_pixels[k].x); //A necessary conversion because the x,y/i,j are driving me nuts
        
        if (diff_img_enlarged.at<uchar>(pixel) == 0)
            continue;

        Mat template_img = Mat::zeros(Size(tfp.diff_img.cols, tfp.diff_img.rows), CV_8U);

        start = high_resolution_clock::now();
        recursion_func(pixel, diff_img_enlarged, 255, template_img);
        //cout << "Recursion Function: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        blob_extractions_time.push_back(duration_cast<microseconds>(high_resolution_clock::now() - start).count());
        
        start = high_resolution_clock::now();
        Mat blob_img;

        bitwise_and(tfp.diff_img, template_img, blob_img); 
        Rect bounding_rect = boundingRect(blob_img);                                                                                                        
        if (bounding_rect.x == 0 || bounding_rect.y == 0 || bounding_rect.x + bounding_rect.width == tfp.diff_img.cols || bounding_rect.y + bounding_rect.height == tfp.diff_img.rows) {
            continue;
        }

        Mat blob_img_cropped = blob_img(bounding_rect);
        Vec2f dir = calculate_direction(blob_img_cropped);
        //cout << "Direction Calculation: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        calc_dir_time.push_back(duration_cast<microseconds>(high_resolution_clock::now() - start).count());

        if (dir.val[0] < 2.0)
        {
            directions.push_back(dir);
            dir_x.push_back(dir.val[0]);
            dir_y.push_back(dir.val[1]);

            Mat element = getStructuringElement(MORPH_RECT,
                Size(3, 3));
            start = high_resolution_clock::now();
            dilate(blob_img, blob_img, element);
            //cout << "Dilation: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
            dilations_time.push_back(duration_cast<microseconds>(high_resolution_clock::now() - start).count());
            templates.push_back(blob_img);

            start = high_resolution_clock::now();
            int angle = floor(atan2(-dir.val[0], dir.val[1]) * 180 / PI);
            if (angle < 0){
                angle += 360;
            }
            angle_per_blob.push_back(angle);
            Mat temp(Size(tfp.diff_img.cols, tfp.diff_img.rows), CV_8UC3, cv::Scalar(angle/2, 255, 255));
            Mat template_img_3d;
            cv::cvtColor(template_img, template_img_3d, cv::COLOR_GRAY2RGB);

            bitwise_and(template_img_3d, temp, temp);
            add(temp, hsv_image, hsv_image);
            cout << "HSV image update: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        }
    }


    cv::destroyWindow("Templates");
    cv::namedWindow("Templates", WINDOW_FREERATIO);
    LOG("VECTOR ANGLES:");
    PRINT_VECTOR(angle_per_blob);
    if (templates.size() > 0) {
        cv::createTrackbar("Template", "Templates", &template_value, templates.size() - 1, [](int, void*) { imshow("Templates", templates[template_value]); });

        std::cout << "Median X " << findMedian(dir_x) << endl;
        std::cout << "Median Y " << findMedian(dir_y) << endl;
        std::cout << "Median Angle: " << atan2(-findMedian(dir_x), findMedian(dir_y)) * 180 / PI << endl;
    }    
    cvtColor(hsv_image, hsv_image, COLOR_HSV2BGR);
    CHECK_IMAGE("HSV", hsv_image, false);

    start = high_resolution_clock::now();

    vector<vector<KeyPoint>> kp_per_template_cur(templates.size()), kp_per_template_prev(templates.size());
    vector<Mat> des_per_template_cur(templates.size()), des_per_template_prev(templates.size());

    for (int i = 0; i < kp_cur.size(); i++) {
        Point2d point = kp_cur[i].pt;
        Mat temp_descriptor;
        for (int j = 0; j < templates.size(); j++) {
            Mat temp = templates[j];
            if (temp.at<uchar>(point) != 0) {
                kp_per_template_cur[j].push_back(kp_cur[i]);
                temp_descriptor.push_back(des_cur.row(i));

                des_per_template_cur[j].push_back(temp_descriptor);
                break;
            }
        }
    }

    for (int i = 0; i < kp_prev.size(); i++) {
        Point2d point = kp_prev[i].pt;
        Mat temp_descriptor;
        for (int j = 0; j < templates.size(); j++) {
            Mat temp = templates[j];
            if (temp.at<uchar>(point) != 0) {
                kp_per_template_prev[j].push_back(kp_prev[i]);
                temp_descriptor.push_back(des_prev.row(i));
                des_per_template_prev[j].push_back(temp_descriptor);
                break;
            }
        }
    }
    std::vector< DMatch > matches;
    Mat match_results, match_results_past;
    for (int i = 0; i < templates.size(); i++) {
        if (kp_per_template_prev[i].size() == 0 || kp_per_template_cur[i].size() == 0)
            continue;
        auto bf = BFMatcher(cv::NORM_HAMMING, true);

        vector<KeyPoint> kp1 = kp_per_template_prev[i];
        vector<KeyPoint> kp2 = kp_per_template_cur[i];

        Mat des1 = des_per_template_prev[i];
        Mat des2 = des_per_template_cur[i];

        //Train: past, Query: present
        bf.match(des1, des2, matches);  
        
        for (DMatch match : matches) {
            int queryId = match.queryIdx;
            int trainId = match.trainIdx;
            cout << "Match: " << queryId << " Matches to: " << trainId << endl;
            KeyPoint point1 = kp1[queryId];
            KeyPoint point2 = kp2[trainId];
            cout << "Respecting keypoints(past): " << point1.pt << " And(present) " << point2.pt << endl;
            Point direction = point2.pt - point1.pt;
            cout << " With Angle: " << atan2(-direction.y, direction.x) * 180 / PI << endl;
        }

        drawMatches(previous, kp1, current, kp2, matches, match_results, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //if (i == 0) match_results_past = match_results.clone();

        CHECK_IMAGE("Matches", match_results, true);
    }

    long long my_method_matching_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();
    cout << "My method of doing correspondences : " << my_method_matching_time  << endl;

    cout << "Mean Recursion Function time: " << MEAN_VALUE(blob_extractions_time) << endl;
    cout << "Mean Calculate Direction: " << MEAN_VALUE(calc_dir_time) << endl;
    cout << "Mean Dilation: " << MEAN_VALUE(dilations_time) << endl;
    cout << "Sum: " << MEAN_VALUE(dilations_time) + MEAN_VALUE(calc_dir_time) + MEAN_VALUE(blob_extractions_time) + my_method_matching_time << endl;

    std::cout << "Blob Number: " << templates.size() << endl;
}

static void frame_trackbar ( int , void* )
{
    std::cout << "petros" << endl;
    if (frame_slider < 2) {
        return;
    }

    std::cout << "Frame trackbar with value: " << frame_slider << std::endl;
    threshold_trackbar(threshold_slider, 0);
    imshow("RGB_Image", images[frame_slider]);
}


int main(int argc, char** argv )
{
    cv::FileStorage fSettings("../TUM1.yaml", cv::FileStorage::READ);
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    ORBextractorLeft = new ORB_SLAM2::ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    std::cout << "OpenCV version : " << CV_VERSION << endl;
    std::cout << "[0,1] angle " << atan2(0, 1) * 180 / PI << endl;
    std::cout << "[1,0] angle " << atan2(1, 0) * 180 / PI << endl;
    std::cout << "[-1,0] angle " << atan2(-1, 0) * 180 / PI << endl;
    //String img_folder_path = argv[1];
    std::vector<String> fn;
    //glob("C:\\Users\\Stagakis\\Desktop\\rgbd_dataset_freiburg1_xyz\\rgb_short", fn, false);
    glob("../rgb/*png", fn, false);
    images.reserve(fn.size());
    for(int i = 0; i < fn.size(); i++){
        Mat img_frame = imread(fn[i]);
        cv::resize(img_frame, img_frame, cv::Size(img_frame.cols * 0.5, img_frame.rows * 0.5), 0, 0, CV_INTER_CUBIC);
        images.push_back(img_frame.clone()); 
    }
    createWindowsAndTrackbars();
    while(1){
        int k = waitKey(0);
        std::cout << k << endl;
        if (k == 97)
        {
            frame_slider--;
            setTrackbarPos("Frame", "Control", frame_slider);
            //frame_trackbar(frame_slider, 0);
        }

        if (k == 100)
        {
            frame_slider++;
            setTrackbarPos("Frame", "Control", frame_slider);
            //frame_trackbar(frame_slider, 0);
        }
        if(k == 27)
            break;
    }

}


void createWindowsAndTrackbars() {

    cv::namedWindow("RGB_Image", WINDOW_FREERATIO);
    moveWindow("RGB_Image", 640, 0);

    cv::namedWindow("Diff", WINDOW_FREERATIO);
    moveWindow("Diff", 0, 0);

    cv::namedWindow("Control", WINDOW_FREERATIO);
    resizeWindow("Control", 640, 480);
    moveWindow("Control", 1280, 0);

    cv::namedWindow("Templates");

    cv::createTrackbar("Frame", "Control", &frame_slider, images.size() - 1, frame_trackbar);
    cv::createTrackbar("Threshold", "Control", &threshold_slider, 255, threshold_trackbar);
}
