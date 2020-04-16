#include <BlobExtractor.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

BlobExtractor::BlobExtractor(cv::Mat _diff_img):diff_img(_diff_img){
    blob_img = cv::Mat::zeros(cv::Size(diff_img.cols, diff_img.rows), CV_8U);
}

void BlobExtractor::ExtractBlobs(){
    //auto start = std::chrono::high_resolution_clock::now();

    cv::Mat diff_img_enlarged = cv::Mat::zeros(cv::Size(3*diff_img.cols, 3*diff_img.rows), CV_8U);
    cv::Rect diff_img_clone_center = cv::Rect(diff_img.cols, diff_img.rows, diff_img.cols, diff_img.rows);
    diff_img.copyTo(diff_img_enlarged(diff_img_clone_center));

    cv::Mat diff_img_white_only;
    cv::threshold(diff_img_enlarged, diff_img_white_only, 254, 255, cv::THRESH_BINARY);

    std::vector<cv::Point> white_pixels;
    cv::findNonZero(diff_img_white_only, white_pixels);

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

        cv::Mat blob_img_cropped = single_blob_img(bounding_rect);

        //Check for validity of blob and append Rect to list if valid
        if(isValid(blob_img_cropped)){
            blob_rects.push_back(bounding_rect);
            blob_img_mask.push_back(binary_mask_image);
            add(single_blob_img, blob_img, blob_img);
            num_of_blobs++;
        }

    }

}
bool BlobExtractor::isValid(cv::Mat& blob){
    int before_points = 0;
    int after_points = 0;
    for (int i = 0; i < blob.rows; i++) {
        for (int j = 0; j < blob.cols; j++) {
            if (blob.at<uchar>(i, j) == 105) {
                before_points++;
            }
            else if (blob.at<uchar>(i, j) == 190) {
                after_points++;
            }
        }
    }
    if (before_points + after_points < PIXEL_THRESHOLD 
    || before_points < 0.5*after_points || after_points < 0.5*before_points
    || before_points == 0 || after_points == 0)
        return false;
    else
        return true;
}
void BlobExtractor::recursion_func(cv::Point pixel, cv::Mat& img, uchar blob_number, cv::Mat& template_img, uchar mcolor){

    if (img.at<uchar>(pixel) == 0)
        return;

    int current_color = img.at<uchar>(pixel);

    if (current_color != 255 && mcolor != 255 && mcolor != current_color)
        return;

    img.at<uchar>(pixel) = 0;
    template_img.at<uchar>(pixel.y - template_img.rows, pixel.x - template_img.cols) = blob_number;

    recursion_func(cv::Point(pixel.x + 0, pixel.y + 1), img, blob_number, template_img, current_color);

    recursion_func(cv::Point(pixel.x + 1, pixel.y + 1), img, blob_number, template_img, current_color);
    recursion_func(cv::Point(pixel.x - 1, pixel.y + 1), img, blob_number, template_img, current_color);

    recursion_func(cv::Point(pixel.x + 0, pixel.y - 1), img, blob_number, template_img, current_color);

    recursion_func(cv::Point(pixel.x + 1, pixel.y - 1), img, blob_number, template_img, current_color);

    recursion_func(cv::Point(pixel.x - 1, pixel.y - 1), img, blob_number, template_img, current_color);


    recursion_func(cv::Point(pixel.x + 1, pixel.y + 0), img, blob_number, template_img, current_color);
    recursion_func(cv::Point(pixel.x - 1, pixel.y + 0), img, blob_number, template_img, current_color);
}

void BlobExtractor::GetBlob(int index, cv::Mat& outImage){
    outImage = blob_img(blob_rects[index]);
}

cv::Mat& BlobExtractor::GetBlobDilated(int index, int dilation_kernel_size){
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(dilation_kernel_size, dilation_kernel_size));

    cv::Rect dilated_rect;
    dilated_rect = blob_rects[index];
    dilated_rect.x -= dilation_kernel_size/2;
    dilated_rect.y -= dilation_kernel_size/2;
    dilated_rect.width += dilation_kernel_size - 1;
    dilated_rect.height += dilation_kernel_size - 1;

    cv::Mat temp;
    cv::dilate(blob_img(dilated_rect), temp, element);

    return temp;
}