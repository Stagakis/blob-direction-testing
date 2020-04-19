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


using namespace cv;
using namespace std;
using namespace std::chrono;


static void threshold_trackbar (int , void* )
{
    LOG("====Threshold trackbar value" << threshold_slider)
    cv::Mat current, previous, pre_previous;
    current = images[frame_slider];
    previous = images[frame_slider - 1];
    pre_previous = images[frame_slider - 2];

    {
        auto start = high_resolution_clock::now();
        cv::cvtColor(current, current_greyscale, cv::COLOR_BGR2GRAY);
        ORBextractorLeft->operator()(current_greyscale, NULL, kp_cur, des_cur);

        cv::cvtColor(previous, previous_greyscale, cv::COLOR_BGR2GRAY);
        ORBextractorLeft->operator()(previous_greyscale, NULL, kp_prev, des_prev);
        cout << "ORB Extraction time*2: " << duration_cast<microseconds>(high_resolution_clock::now() - start).count() << endl;

        start = high_resolution_clock::now();
        auto bf = BFMatcher(cv::NORM_HAMMING, true);
        std::vector< DMatch > matches;
        bf.match(des_prev, des_cur, matches);
        Mat match_results_whole_image_orb;
        drawMatches(previous, kp_prev, current, kp_cur, matches, match_results_whole_image_orb, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cout << "Calculating all matches from brute force: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        CHECK_IMAGE(match_results_whole_image_orb, false);
    }
    
    LOG("=====START OF PROCESSING========");
    LOG("Processing three frames");
    tfp = new ThreeFrameProcesser(current, previous, pre_previous);
    tfp->calculateDifferences(threshold_slider);
    diff_image = tfp->diff_img;
    cv::Mat visible_parts;
    tfp->calculateVisibleParts(visible_parts);
    
    Mat hsv_image = Mat::zeros(Size(diff_image.cols, diff_image.rows), CV_8UC3);   

    LOG("Extracting Blobs");
    blextr = new BlobExtractor(diff_image, tfp->diff_cur_prev, tfp->diff_prev_preprev);
    blextr->ExtractBlobs();
    LOG("Blobs found from Extraction: " << blextr->num_of_blobs)

    LOG("Iterating over Blobs");
    templates.clear();
    angle_per_blob.clear();
    list_of_match_results.clear();
    list_of_match_results_withcrosscheck.clear();
    for (int k = 0; k < blextr->num_of_blobs; k++){
        LOG("==K is " << k << " Out of " << blextr->num_of_blobs);

        templates.push_back(blextr->blob_img_mask[k]);

        cv::Mat blob_img_mask_dilated;
        cv::Mat match_results;
        vector<KeyPoint> kp_cur_blob, kp_prev_blob;
        Mat des_cur_blob, des_prev_blob;
        std::vector< DMatch > matches;

        auto start = high_resolution_clock::now();
        blextr->GetBlobDilated(k, blob_img_mask_dilated, 2*dilation_slider+1);
        cout << "TIME Dilate Duration: " << duration_cast<microseconds>(high_resolution_clock::now() - start).count() << endl;

        filter_keypoints(kp_cur, des_cur, kp_cur_blob, des_cur_blob, 190, blob_img_mask_dilated);
        filter_keypoints(kp_prev, des_prev, kp_prev_blob, des_prev_blob, 105, blob_img_mask_dilated);
        if (kp_cur_blob.size() == 0 || kp_prev_blob.size() == 0)
            continue;

        cv::Mat blob_img_bb;
        blextr->GetBlob(k, blob_img_bb);
        angle_per_blob.push_back(calculate_angle_by_com(blob_img_bb));

        update_hsv_image(hsv_image, angle_per_blob.back(), blextr->blob_img_mask[k]);

        if(kp_prev_blob.size() == 0 || kp_cur_blob.size() == 0) LOG("PEEEEEEEEEETROOOOOOOOOOOOOO!!!!!!!!!!")

        auto bf = BFMatcher().create(cv::NORM_HAMMING, true);
        start = high_resolution_clock::now();
        bf->match(des_prev_blob, des_cur_blob, matches);
        cout << "TIME With Crosscheck (no Mask): " << duration_cast<microseconds>(high_resolution_clock::now() - start).count() << endl;
        drawMatches(previous, kp_prev_blob, current, kp_cur_blob, matches, match_results);//, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        list_of_match_results_withcrosscheck.push_back(match_results.clone());

        bf = BFMatcher().create(cv::NORM_HAMMING, false);
        cv::Mat mask_mat;
        start = high_resolution_clock::now();
        create_mask_mat(mask_mat, kp_cur_blob, kp_prev_blob, angle_per_blob.back(), blob_angle_tolerance);
        bf->match(des_prev_blob, des_cur_blob, matches, mask_mat); 
        cout << "TIME Without Crosscheck (Mask): " << duration_cast<microseconds>(high_resolution_clock::now() - start).count() << endl;

        drawMatches(previous, kp_prev_blob, current, kp_cur_blob, matches, match_results);//, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        list_of_match_results.push_back(match_results.clone());
    }
    //LOG("After drawing matches: ") 

    std::cout << "Blob Number: " << templates.size() << endl;

    //---------------WINDOW SHOWING ------------------------------//
    calc_optical_flow_gt(previous, current);
    cv::imshow("Diff", diff_image);
    cv::namedWindow("VisiblePart1", WINDOW_FREERATIO);
    cv::imshow("VisiblePart1", visible_parts);
    //cv::destroyWindow("Templates");
    cv::namedWindow("Templates", WINDOW_FREERATIO);
    if (templates.size() > 1) {
        cv::createTrackbar("Template", "Templates", &template_value, templates.size() - 1, [](int, void*) -> void { cv::Mat temp; bitwise_and(templates[template_value], diff_image, temp); cout <<"Blob Angle: " << angle_per_blob[template_value] << endl; imshow("Templates", temp); });
    }    

    //cv::destroyWindow("List_of_matches");
    cv::namedWindow("List_of_matches", WINDOW_FREERATIO);
    cv::namedWindow("List_of_matches_withcrosscheck", WINDOW_FREERATIO);

    if (list_of_match_results.size() > 1) {
        cv::createTrackbar("Match", "Control", &match_result_value, list_of_match_results.size() - 1, 
        [](int, void*) -> void { 
            imshow("List_of_matches", list_of_match_results[match_result_value]);
            imshow("List_of_matches_withcrosscheck", list_of_match_results_withcrosscheck[match_result_value]); 
            });
        cv::imshow("List_of_matches", list_of_match_results[match_result_value]);
        cv::imshow("List_of_matches_withcrosscheck", list_of_match_results_withcrosscheck[match_result_value]);
    }    


    cvtColor(hsv_image, hsv_image, COLOR_HSV2BGR);
    CHECK_IMAGE(hsv_image, false);

}

static void frame_trackbar ( int , void* )
{
    if (frame_slider < 2) {
        return;
    }

    std::cout << "===Frame trackbar with value: " << frame_slider << std::endl;
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
    LOG("Loading images");
    for(int i = 0; i < fn.size(); i++){
        Mat img_frame = imread(fn[i]);
        cv::resize(img_frame, img_frame, cv::Size(img_frame.cols * 0.5, img_frame.rows * 0.5), 0, 0, CV_INTER_CUBIC);
        images.push_back(img_frame.clone()); 
    }
    LOG("Creating Windows");
    createWindowsAndTrackbars();
    LOG("Starting Mainloop");
    frame_trackbar(0, 0);
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

    cv::namedWindow("Templates", WINDOW_FREERATIO);
    cv::namedWindow("List_of_matches", WINDOW_FREERATIO);
    cv::namedWindow("List_of_matches_withcrosscheck", WINDOW_FREERATIO);

    cv::createTrackbar("Frame", "Control", &frame_slider, images.size() - 1, frame_trackbar);
    cv::createTrackbar("Threshold", "Control", &threshold_slider, 255, threshold_trackbar);
    cv::createTrackbar("Dilation(2*X+1)", "Control", &dilation_slider, 3, [](int, void*) -> void {threshold_trackbar(0,0); });
    cv::createTrackbar("Angle_tolerance(2*X)", "Control", &blob_angle_tolerance, 15, [](int, void*) -> void {threshold_trackbar(0,0); });

}

