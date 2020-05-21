#include <BlobExtractor.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#define GET_VARIABLE_NAME(Variable) (#Variable)
#define CHECK_IMAGE(mat_name, wait) cv::namedWindow(GET_VARIABLE_NAME(mat_name), cv::WINDOW_KEEPRATIO); cv::imshow(GET_VARIABLE_NAME(mat_name), mat_name); if(wait) cv::waitKey(0);
/*
BlobExtractor::BlobExtractor(const cv::Mat& _diff_img,const cv::Mat& _diff_img_cur_prev,const cv::Mat& _diff_img_prev_preprev):
diff_img(_diff_img), 
diff_img_cur_prev(_diff_img_cur_prev),
diff_img_prev_preprev(_diff_img_prev_preprev)
{
    num_of_blobs = 0;
}
*/
BlobExtractor::BlobExtractor()
{
    num_of_blobs = 0;
    blob_img_full.reserve(500);
    blob_rects.reserve(500);
}
/*
void BlobExtractor::Downscale(){
    scale_factor = 4;
    cv::resize(diff_img, diff_img, cv::Size(), 1.0/scale_factor, 1.0/scale_factor, CV_INTER_NN);
    //cv::resize(unfiltered_blob_img, unfiltered_blob_img, cv::Size(), 1.0/scale_factor, 1.0/scale_factor, CV_INTER_NN);
    cv::resize(diff_img_cur_prev, diff_img_cur_prev, cv::Size(), 1.0/scale_factor, 1.0/scale_factor, CV_INTER_NN);
    cv::resize(diff_img_prev_preprev, diff_img_prev_preprev, cv::Size(), 1.0/scale_factor,1.0/scale_factor, CV_INTER_NN);
}
*/

void mfindNonZero(cv::Mat& image, std::vector<cv::Point>& white_pixels){
    for (int i = 0; i < image.rows; i++)  //row first!
    {
        uchar* data= image.ptr(i);

        for (int j = 0; j < image.cols; j++)
        {
            if(data[j] == 255){
                white_pixels.emplace_back(j , i);
            }
        }
    }
}

#include <chrono>
std::chrono::steady_clock::time_point t;
std::chrono::steady_clock::time_point t_other;
#define TIME2(name, x) x; //t = std::chrono::steady_clock::now(); x ; std::cout << name << ": " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t).count() << std::endl
#define TIME3(name, x) x;// t_other = std::chrono::steady_clock::now(); x ; std::cout << name << ": " << std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_other).count() << std::endl
#define TIME_ACCUM(Variable, x) x; //std::chrono::steady_clock::time_point t_accum_ = std::chrono::steady_clock::now(); x ; Variable += std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_accum_).count()


bool BlobExtractor::GetNextBlob(cv::Mat& out_blob_img, cv::Rect& out_bb) {
    auto timing = std::chrono::steady_clock::now();
    TIME2("--WholeOfGetNextBlob  ",
    bool found_blob = false; //TODO remove it in the end, it's only for debug
    for(int i = start; i < white_pixels.size(); i++) {
        cv::Point pixel = white_pixels[i]; // Point(white_pixels[k].y, white_pixels[k].x); //A necessary conversion because the x,y/i,j are driving me nuts

        if (diff_img.at<uchar>(pixel) == 0)
            continue;

        cv::Mat binary_mask_image = cv::Mat::zeros(cv::Size(diff_img.cols, diff_img.rows), CV_8U);

        int before_points = 0;
        int after_points = 0;
        cv::Point top_left(diff_img.cols, diff_img.rows);
        cv::Point bottom_right(0, 0);

        recursion_func2(pixel, diff_img, 255, binary_mask_image, before_points, after_points, top_left, bottom_right);

        if (!(   before_points + after_points < PIXEL_THRESHOLD / (4)
              || before_points < 0.3 * after_points || after_points < 0.3 * before_points
              || before_points == 0 || after_points == 0)) {

            out_bb = cv::Rect(top_left, bottom_right);

            /*
            if(out_bb.width > 15)
                cv::imwrite("/home/stagakis/ORB_SLAM2/debug_folder/_Blob_width" + std::to_string(time(0)) + "_" + std::to_string(num_of_blobs) +".png",binary_mask_image);
            else if( out_bb.height > 15)
                cv::imwrite("/home/stagakis/ORB_SLAM2/debug_folder/_Blob_height" + std::to_string(time(0)) + "_" + std::to_string(num_of_blobs) +".png",binary_mask_image);
            */

            out_blob_img = binary_mask_image;
            start = i;
            found_blob = true;
            num_of_blobs++;
            break;
        }

    }
    );
    blob_extraction_total_time += std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - timing).count();
    return found_blob;
}

void BlobExtractor::ExtractBlobs(cv::Mat& diff_img) {

    TIME2("--FindingNonZero      ",
    std::vector<cv::Point> white_pixels;
    mfindNonZero(diff_img, white_pixels);
    );

    cv::Mat binary_mask_image;

    //std::cout << "White pixel extraction: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;
    TIME2("--ActualLoop          ",
    for (int k = 0; k < white_pixels.size(); k++) {

        cv::Point pixel = white_pixels[k]; // Point(white_pixels[k].y, white_pixels[k].x); //A necessary conversion because the x,y/i,j are driving me nuts

        if (diff_img.at<uchar>(pixel) == 0)
            continue;

        binary_mask_image = cv::Mat::zeros(cv::Size(diff_img.cols, diff_img.rows), CV_8U);

        int before_points = 0;
        int after_points=0;
        cv::Point top_left(diff_img.cols, diff_img.rows);
        cv::Point bottom_right(0, 0);

        TIME3("--RecursiveFunc       ", recursion_func2(pixel, diff_img, 255, binary_mask_image, before_points, after_points, top_left, bottom_right););

        if(!(before_points + after_points < PIXEL_THRESHOLD / (4)
             || before_points < 0.3 * after_points || after_points < 0.3 * before_points
             || before_points == 0 || after_points == 0)){
            //TIME3("--Sparsification:     ",cv::SparseMat temp(binary_mask_image););
            //cv::SparseMat temp2(binary_mask_image);

            cv::Rect bb(top_left, bottom_right);
            TIME3("--PushBack            ",blob_img_full.push_back(binary_mask_image.clone()););
            blob_rects.push_back(bb);
            num_of_blobs++;

        }

        /*/

        //start = std::chrono::high_resolution_clock::now();

        cv::Mat single_blob_img;
        bitwise_and(diff_img, binary_mask_image, single_blob_img);

        cv::Rect bounding_rect = boundingRect(single_blob_img);

        if (bounding_rect.x == 0 || bounding_rect.y == 0 || bounding_rect.x + bounding_rect.width == diff_img.cols || bounding_rect.y + bounding_rect.height == diff_img.rows) {
            continue;
        }

        //add(single_blob_img, unfiltered_blob_img, unfiltered_blob_img);
        //Check for validity of blob and append Rect to list if valid
        if(isValid(single_blob_img(bounding_rect))){
            //blob_rects.push_back(bounding_rect);
            //blob_img_mask.push_back(binary_mask_image);
            blob_img_full.push_back(single_blob_img);
            num_of_blobs++;
        }
        */

    });

}

/*
void BlobExtractor::ExtractBlobs(){
    //auto start = std::chrono::high_resolution_clock::now();

    cv::Mat diff_img_enlarged = cv::Mat::zeros(cv::Size(3*diff_img.cols, 3*diff_img.rows), CV_8U);
    cv::Rect diff_img_clone_center = cv::Rect(diff_img.cols, diff_img.rows, diff_img.cols, diff_img.rows);
    diff_img.copyTo(diff_img_enlarged(diff_img_clone_center));


    std::vector<cv::Point> white_pixels;
    mfindNonZero(diff_img_enlarged, white_pixels);

    //std::cout << "White pixel extraction: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

    for (int k = 0; k < white_pixels.size(); k++) {
            
        cv::Point pixel = white_pixels[k]; // Point(white_pixels[k].y, white_pixels[k].x); //A necessary conversion because the x,y/i,j are driving me nuts
        
        if (diff_img_enlarged.at<uchar>(pixel) == 0)
            continue;

        cv::Mat binary_mask_image = cv::Mat::zeros(cv::Size(diff_img.cols, diff_img.rows), CV_8U);

        //start = std::chrono::high_resolution_clock::now();
        recursion_func(pixel, diff_img_enlarged, 255, binary_mask_image);
        //blob_extractions_time.push_back(duration_cast<microseconds>(high_resolution_clock::now() - start).count());
        
        //start = std::chrono::high_resolution_clock::now();

        cv::Mat single_blob_img;
        bitwise_and(diff_img, binary_mask_image, single_blob_img); 

        cv::Rect bounding_rect = boundingRect(single_blob_img);        

        if (bounding_rect.x == 0 || bounding_rect.y == 0 || bounding_rect.x + bounding_rect.width == diff_img.cols || bounding_rect.y + bounding_rect.height == diff_img.rows) {
            continue;
        }

        //add(single_blob_img, unfiltered_blob_img, unfiltered_blob_img);
        //Check for validity of blob and append Rect to list if valid
        if(isValid(single_blob_img(bounding_rect))){
            //blob_rects.push_back(bounding_rect);
            //blob_img_mask.push_back(binary_mask_image);
            blob_img_full.push_back(single_blob_img);
            num_of_blobs++;
        }

    }

}
 */
bool BlobExtractor::isValid(const cv::Mat& blob){
    int before_points = 0;
    int after_points = 0;
    for (int i = 0; i < blob.rows; i++) {
        for (int j = 0; j < blob.cols; j++) {
            if (blob.at<uchar>(i, j) == THRESHOLD_VALUE_PREV) {
                before_points++;
            }
            else if (blob.at<uchar>(i, j) == THRESHOLD_VALUE_CUR) {
                after_points++;
            }
        }
    }
    return !(before_points + after_points < PIXEL_THRESHOLD / (4)
             || before_points < 0.3 * after_points || after_points < 0.3 * before_points
             || before_points == 0 || after_points == 0);
}

void BlobExtractor::recursion_func2(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img,
                                   int& before_points, int& after_points,
                                   cv::Point &top_left, cv::Point& bottom_right, uchar mcolor) {


    int current_color;

    if(pixel.inside(cv::Rect(0,0,img.cols, img.rows))) current_color = img.at<uchar>(pixel);
    else return;


    if ( (current_color != 255 && mcolor != 255 && mcolor != current_color) || current_color == 0)
        return;

    switch (current_color) {
        case THRESHOLD_VALUE_CUR:
            ++after_points;
            break;
        case THRESHOLD_VALUE_PREV:
            ++before_points;
            break;
    }

    if(pixel.x < top_left.x) top_left.x = pixel.x;
    if(pixel.y < top_left.y) top_left.y = pixel.y;
    if(pixel.x > bottom_right.x) bottom_right.x = pixel.x;
    if(pixel.y > bottom_right.y) bottom_right.y = pixel.y;


    img.at<uchar>(pixel) = 0;
    template_img.at<uchar>(pixel) = current_color;


    recursion_func2(cv::Point(pixel.x + 0, pixel.y + 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    //recursion_func2(cv::Point(pixel.x + 1, pixel.y + 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    //recursion_func2(cv::Point(pixel.x - 1, pixel.y + 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    recursion_func2(cv::Point(pixel.x + 0, pixel.y - 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    //recursion_func2(cv::Point(pixel.x + 1, pixel.y - 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    //recursion_func2(cv::Point(pixel.x - 1, pixel.y - 1), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    recursion_func2(cv::Point(pixel.x + 1, pixel.y + 0), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);
    recursion_func2(cv::Point(pixel.x - 1, pixel.y + 0), img, blob_number, template_img, before_points, after_points,top_left, bottom_right, current_color);

}


void BlobExtractor::recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img,
        int& before_points, int& after_points, uchar mcolor) {


    int current_color;

    if(pixel.inside(cv::Rect(0,0,img.cols, img.rows))) current_color = img.at<uchar>(pixel);
    else return;


    if ( (current_color != 255 && mcolor != 255 && mcolor != current_color) || current_color == 0)
        return;

    switch (current_color) {
        case THRESHOLD_VALUE_CUR:
            ++after_points;
            break;
        case THRESHOLD_VALUE_PREV:
            ++before_points;
            break;
    }

    img.at<uchar>(pixel) = 0;
    template_img.at<uchar>(pixel) = current_color;



    recursion_func(cv::Point(pixel.x + 0, pixel.y + 1), img, blob_number, template_img, before_points, after_points, current_color);

    recursion_func(cv::Point(pixel.x + 1, pixel.y + 1), img, blob_number, template_img, before_points, after_points, current_color);
    recursion_func(cv::Point(pixel.x - 1, pixel.y + 1), img, blob_number, template_img, before_points, after_points, current_color);

    recursion_func(cv::Point(pixel.x + 0, pixel.y - 1), img, blob_number, template_img, before_points, after_points, current_color);

    recursion_func(cv::Point(pixel.x + 1, pixel.y - 1), img, blob_number, template_img, before_points, after_points, current_color);

    recursion_func(cv::Point(pixel.x - 1, pixel.y - 1), img, blob_number, template_img, before_points, after_points, current_color);


    recursion_func(cv::Point(pixel.x + 1, pixel.y + 0), img, blob_number, template_img, before_points, after_points, current_color);
    recursion_func(cv::Point(pixel.x - 1, pixel.y + 0), img, blob_number, template_img, before_points, after_points, current_color);

}

void BlobExtractor::GetBlobFullSize(int index, cv::Mat& outImage){
    outImage = blob_img_full[index];
}

BlobExtractor::BlobExtractor(cv::Mat &_diff_img) {
    diff_img = _diff_img.clone();
    mfindNonZero(diff_img, white_pixels);
    start = 0;
    num_of_blobs = 0;
    blob_extraction_total_time = 0;
}


/*
void BlobExtractor::GetBlobDilated(int index, cv::Mat& outImage, int dilation_kernel_size){

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(dilation_kernel_size, dilation_kernel_size));


    cv::Rect dilated_rect;
    int offset = dilation_kernel_size/2;
    dilated_rect = blob_rects[index];
    dilated_rect.x -= offset;
    dilated_rect.y -= offset;
    dilated_rect.width += 2*offset;
    dilated_rect.height += 2*offset;


    cv::Mat mask1, mask2;
    cv::bitwise_and(diff_img_cur_prev,  blob_img_mask[index], mask1);
    cv::dilate(mask1(dilated_rect), mask1(dilated_rect), element);
    //CHECK_IMAGE("diff_img_cur_prev", mask1, false);

    cv::bitwise_and(diff_img_prev_preprev,  blob_img_mask[index], mask2);
    cv::dilate(mask2(dilated_rect), mask2(dilated_rect), element);
    //CHECK_IMAGE("diff_img_prev_preprev", mask2, false);

    cv::add(mask1, mask2, outImage);
    //CHECK_IMAGE("DILATEDOUTIMAGE", outImage, true);

}
*/