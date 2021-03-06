#include "detector.h"
#include <thread>
#include "utils.hpp"

void Detect(){
    Detector* detector = new Detector;
    string xml_path = "../res/best.xml";
    detector->init(xml_path, 0.5, 0.5);

    /*
    VideoCapture capture;
    capture.open(0);
    Mat src;
    while(1){
        capture >> src;
        vector<Detector::Object> detected_objects;
    detector->process_frame(src,detected_objects);
    for(int i=0;i<detected_objects.size();++i){
        int xmin = detected_objects[i ].rect.x;
        int ymin = detected_objects[i].rect.y; 
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(src, rect, Scalar(255, 0, 0),1, LINE_8,0);
    }
        imshow("cap",src);
        waitKey(1);
    }
    */
    Mat src = imread("../res/496.jpg");
    Mat osrc = src.clone();
    resize(osrc,osrc,Size(640,640));
    vector<Detector::Object> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    detector->process_frame(src,detected_objects);
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"use "<<diff.count()<<" s" << endl;
    for(size_t i=0;i<detected_objects.size();++i){
        int xmin = detected_objects[i].rect.x;
        int ymin = detected_objects[i].rect.y;
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
        cv::rectangle(osrc, rect, Scalar(255, 0, 255),2, cv::LINE_AA,0);
        cv::putText(osrc,detected_objects[i].name,
            cv::Point(xmin,ymin - 10),
            cv::FONT_HERSHEY_COMPLEX_SMALL,
            1,cv::Scalar(255,0,255),
            2,
            cv::LINE_AA);
    }
    cv::line(osrc,cv::Point(osrc.cols * 0.5, 0),cv::Point(osrc.cols * 0.5, osrc.rows),
            cv::Scalar(255,0,100),2,cv::LINE_8);

    std::sort(detected_objects.begin(), detected_objects.end());
    cv::rectangle(
            osrc,
            detected_objects.at(0).rect,
            cv::Scalar(0, 0, 0),
            2,
            cv::LINE_8,
            0);
    cv::line(osrc,cv::Point(osrc.cols * 0.5, 0),cv::Point(osrc.cols * 0.5, osrc.rows),
            cv::Scalar(255,0,100),2,cv::LINE_8);
    imshow("result",osrc);
    waitKey(0);
}
int main(int argc, char const *argv[])
{
    Detect();
}
