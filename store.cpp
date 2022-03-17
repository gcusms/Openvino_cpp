    
    // Detector* detector = new Detector;
    // string xml_path = "/home/wolf/Desktop/temp2/demo/res/best.xml";
    // detector->init(xml_path,0.3,0.5);
    // VideoCapture capture;
    // capture.open("/home/wolf/Desktop/temp2/demo/res/block_v.mp4");
    // Mat src;
    // try
    // {
    //     while(1){
    //     capture >> src;
    //     auto start = chrono::high_resolution_clock::now();
    //     vector<Detector::Object> detected_objects;
    //     auto end = chrono::high_resolution_clock::now();
        // std::chrono::duration<double> diff = end - start;
        // cout<<"use "<<diff.count()<<" s" << endl;
    //     detector->process_frame(src,detected_objects);
    //     for(int i=0;i<detected_objects.size();++i){
    //     int xmin = detected_objects[i].rect.x;
    //     int ymin = detected_objects[i].rect.y;
    //     int width = detected_objects[i].rect.width;
    //     int height = detected_objects[i].rect.height;
    //     Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
    //     cv::rectangle(src, rect, Scalar(200,0,200),2, LINE_AA,0);
    //     cv::putText(src,detected_objects[i].name + " "+to_string(detected_objects[i].status),cv::Point(rect.x,rect.y - 5),1,1,Scalar(200, 0, 255),1,LINE_4);
    //     }
    //         imshow("cap",src);
    //         waitKey(1);
    //     }
    // }
    // catch(const std::exception& e)
    // {
    //     std::cerr << e.what() << '\n';
    // }

    // // Mat src = imread("/home/wolf/Desktop/temp2/demo/res/13.jpg");
    // // Mat osrc = src.clone();
    // // resize(osrc,osrc,Size(640,640));
    // // vector<Detector::Object> detected_objects;
    // // auto start = chrono::high_resolution_clock::now();
    // // detector->process_frame(src,detected_objects);
    // // auto end = chrono::high_resolution_clock::now();
    // // std::chrono::duration<double> diff = end - start;
    // // cout<<"use "<<diff.count()<<" s" << endl;
    // // for(int i=0;i<detected_objects.size();++i){
    // //     int xmin = detected_objects[i].rect.x;
    // //     int ymin = detected_objects[i].rect.y;
    // //     int width = detected_objects[i].rect.width;
    // //     int height = detected_objects[i].rect.height;
    // //     Rect rect(xmin, ymin, width, height);//左上坐标（x,y）和矩形的长(x)宽(y)
    //     // cv::rectangle(osrc, rect, Scalar(rand()%200,rand()%200,rand()%200),2, LINE_AA,0);
    //     // cv::putText(osrc,detected_objects[i].name + " "+to_string(detected_objects[i].status),cv::Point(rect.x,rect.y - 5),1,1,Scalar(200, 0, 255),1,LINE_4);
    // // }  
    // // imwrite("../result1.jpg",osrc);
    // // imshow("result",osrc);
    // waitKey(0);