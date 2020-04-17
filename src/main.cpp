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
        Mat match_results;
        drawMatches(previous, kp_prev, current, kp_cur, matches, match_results, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cout << "Calculating all matches from brute force: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << endl;
        CHECK_IMAGE("WholeImageBruteForce", match_results, false);
    }

    ThreeFrameProcesser tfp(current, previous, pre_previous);
    tfp.calculateDifferences(threshold_slider);
    diff_image = tfp.diff_img;
    cv::Mat visible_parts;
    tfp.calculateVisibleParts(visible_parts);
    
    Mat hsv_image = Mat::zeros(Size(tfp.diff_img.cols, tfp.diff_img.rows), CV_8UC3);   

    BlobExtractor blextr(tfp.diff_img);
    LOG("Extracting Blobs");
    blextr.ExtractBlobs();

    LOG("Iterating over Blobs");
    templates.clear();
    list_of_match_results.clear();
    for (int k = 0; k < blextr.num_of_blobs; k++){
        vector<KeyPoint> kp_cur_blob, kp_prev_blob;
        Mat des_cur_blob, des_prev_blob;
        std::vector< DMatch > matches;

        templates.push_back(blextr.blob_img_mask[k]);
        LOG("FILTERING KEYPONTS")
        filter_keypoints(kp_cur, des_cur, kp_cur_blob, des_cur_blob, 190, blextr.blob_img_mask[k]);
        filter_keypoints(kp_prev, des_prev, kp_prev_blob, des_prev_blob, 105, blextr.blob_img_mask[k]);
        if (kp_cur_blob.size() == 0 || kp_prev_blob.size() == 0)
            continue;

        cv::Mat blob_img;
        blextr.GetBlob(k, blob_img);
        Vec2f dir = calculate_direction2(blob_img);
 
        int angle = floor(atan2(-dir.val[0], dir.val[1]) * 180 / PI);
        if (angle < 0) angle += 360;
        cout << "Direction(angle) of Blob No. " << k << " is " << angle << endl;;
        update_hsv_image(hsv_image, angle, blextr.blob_img_mask[k]);

        if(kp_prev_blob.size() == 0 || kp_cur_blob.size() == 0)
            LOG("PEEEEEEEEEETROOOOOOOOOOOOOO!!!!!!!!!!")

        cv::Mat mask_mat = cv::Mat::zeros(Size(kp_prev_blob.size(), kp_cur_blob.size()), CV_8UC1);

        for(int i = 0; i < mask_mat.rows; i++){
            for (int j = 0 ; j< mask_mat.cols; j++){
                Point from = kp_prev_blob[i].pt;
                Point to = kp_cur_blob[j].pt;
                Vec2f keypoint_dir = Vec2f(to.y - from.y, to.x - from.x) ;
                int keypont_angle = floor(atan2(-keypoint_dir.val[0], keypoint_dir.val[1]) * 180 / PI);
                if (keypont_angle < 0) keypont_angle += 360;

                if( abs(keypont_angle - angle) < 20 )
                {
                    mask_mat.at<uchar>(i,j) = 1;
                    cout << "Keypont(angle) "<< " is " << keypont_angle <<endl;
                }
            }
        }
        mask_mat = mask_mat.t();
        LOG("Size mask: ");
        LOG(mask_mat.size());
        LOG("QuerryDrecriptorSize :");
        LOG(des_prev_blob.size());
        LOG("TrainDrecriptorSize :");
        LOG(des_cur_blob.size());
        auto bf = BFMatcher().create(cv::NORM_HAMMING, false);
        //Train: past, Query: present
        bf->match(des_prev_blob, des_cur_blob, matches, mask_mat);  
        cv::Mat match_results;
        //LOG("Size kp_prev_blob: ");
        //LOG(kp_prev_blob.size());
        //LOG("Size kp_cur_blob: ");
        //LOG(kp_cur_blob.size());
        //LOG("Size matches: ");
        //LOG(matches.size());
        drawMatches(previous, kp_prev_blob, current, kp_cur_blob, matches, match_results, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //CHECK_IMAGE("Matches", match_results, true);
        list_of_match_results.push_back(match_results);
    }


    std::cout << "Blob Number: " << templates.size() << endl;

    //---------------WINDOW SHOWING ------------------------------//
    cv::imshow("Diff", tfp.diff_img);
    cv::namedWindow("VisiblePart1", WINDOW_FREERATIO);
    cv::imshow("VisiblePart1", visible_parts);
    cv::destroyWindow("Templates");
    cv::namedWindow("Templates", WINDOW_FREERATIO);
    if (templates.size() > 1) {
        cv::createTrackbar("Template", "Templates", &template_value, templates.size() - 1, [](int, void*) -> void { cv::Mat temp; bitwise_and(templates[template_value], diff_image, temp); imshow("Templates", temp); });
    }    
    cv::destroyWindow("List_of_matches");
    cv::namedWindow("List_of_matches", WINDOW_FREERATIO);
    if (list_of_match_results.size() > 1) {
        cv::createTrackbar("Match", "List_of_matches", &match_result_value, list_of_match_results.size() - 1, [](int, void*) -> void { imshow("List_of_matches", list_of_match_results[match_result_value]); });
    }    
    cvtColor(hsv_image, hsv_image, COLOR_HSV2BGR);
    CHECK_IMAGE("HSV", hsv_image, false);

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
    LOG("Loading images");
    for(int i = 0; i < fn.size(); i++){
        Mat img_frame = imread(fn[i]);
        cv::resize(img_frame, img_frame, cv::Size(img_frame.cols * 0.5, img_frame.rows * 0.5), 0, 0, CV_INTER_CUBIC);
        images.push_back(img_frame.clone()); 
    }
    LOG("Creating Windows");
    createWindowsAndTrackbars();
    LOG("Starting Mainloop");
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
    cv::namedWindow("List_of_matches");

    cv::createTrackbar("Frame", "Control", &frame_slider, images.size() - 1, frame_trackbar);
    cv::createTrackbar("Threshold", "Control", &threshold_slider, 255, threshold_trackbar);
}
