#include "infer/detector.h"
#include "devices/camera/mv_video_capture.hpp"
#include <chrono>

int main(int argc, char const *argv[])
{
    
    int camera_exposure = 5000;
    mindvision::CameraParam camera_params(0, mindvision::RESOLUTION_1280_X_1024,
                                    camera_exposure);
    mindvision::VideoCapture *mv_capture = new mindvision::VideoCapture(camera_params);
    // auto mv_capture = std::make_shared<mindvision::VideoCapture>(camera_params);
    cv::Mat src_img;

    while (true)
    {
        if (mv_capture->isindustryimgInput())
        {      
            mv_capture->cameraReleasebuff();
            src_img = mv_capture->image();
            auto start = std::chrono::high_resolution_clock::now();
            if(!src_img.empty()) {
                cv::imshow("FRAME",src_img);
                cv::waitKey(1);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            fmt::print("use:{}s",diff.count());
        }
    }
    
}
