 
#include<iostream>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <inference_engine.hpp>
#include <cmath>
 
using namespace std;
// using namespace InferenceEngine;
using namespace cv;
using namespace cv::dnn;
 
string xml = "/home/wolf/Desktop/temp2/demo/res/best.xml";
string bin = "/home/wolf/Desktop/temp2/demo/res/best.bin";
 
 
int main() {
    Mat src=cv::imread("/home/wolf/Desktop/temp2/demo/res/zidane.jpg");
    Net net = readNetFromModelOptimizer(xml, bin);
    net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);//使用openvino作为推理引擎
    net.setPreferableTarget(DNN_TARGET_CPU);
    Mat blob = blobFromImage(src, 1.0, Size(640, 640), Scalar(), true, false, 5);
    net.setInput(blob);
    float confidenceThreshold = 0.1;
    Mat detection = net.forward();
    vector<double> layerTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layerTimings) / freq;
    cout<<"openvino模型推理时间为:"<<time<<" ms"<<endl;
 
    int h = src.size().height;
    int w = src.size().width;
    cv::Mat dectetionMat(detection.size[2], detection.size[3], CV_32F,detection.ptr<float>());
    for (int i = 0; i < dectetionMat.rows;i++) {
        float confidence = dectetionMat.at<float>(i, 2);
        // cout << confidence << endl;
        if (confidence> confidenceThreshold) {      
            int idx= dectetionMat.at<float>(i, 1);
            // cout << "idx is " << idx << endl;
            int left= static_cast<int>(dectetionMat.at<float>(i, 3) * w);
            int top = static_cast<int>(dectetionMat.at<float>(i, 4) * h);
            int right = static_cast<int>(dectetionMat.at<float>(i, 5) * w);
            int bottom = static_cast<int>(dectetionMat.at<float>(i, 6) * h);
            cv::rectangle(src,Rect(left,top,right-left,bottom-top),Scalar(255,0,0),2);            
        }
    }
    cv::imwrite("/home/wolf/Desktop/temp2/demo/res/output.jpg",src);
    cv::imshow("3.jpg",src);
    cv::waitKey(0);
    return 0;
}