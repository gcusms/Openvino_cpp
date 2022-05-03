#include "detector.h"

Detector::Detector(){}

Detector::~Detector(){}
string names2[] = {"lie","erect"};
//注意此处的阈值是框和物体prob乘积的阈值
bool Detector::parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
    vector<Rect>& o_rect,vector<float>& o_rect_cof,vector<int>& label_input){
    vector<int> anchors = get_anchors(net_grid);
   LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
   const float *output_blob = blobMapped.as<float *>();
   //80 个类是 85,一个类是6,n个类是n+5
   //int item_size = 6;
   int item_size = 7;
    size_t anchor_n = 3;
    std::cout << "net_grid: " << net_grid << std::endl;
    for(int n=0;n<anchor_n;++n)
        for(int i=0;i<net_grid;++i)
            for(int j=0;j<net_grid;++j)
            {
                double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4];
                box_prob = sigmoid(box_prob);
                //框置信度不满足则整体置信度不满足
                if(box_prob < cof_threshold)
                    continue;
                
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 3];

                double max_prob = 0;
                int idx=0;
                for(int t=5;t<item_size;++t){
                    double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                    tp = sigmoid(tp);
                    if(tp > max_prob){
                        max_prob = tp;
                        idx = t;
                    }
                }
                float cof = box_prob * max_prob;                
                //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if(cof < cof_threshold)
                    continue;

                x = (sigmoid(x)*2 - 0.5 + j)*640.0f/net_grid;
                y = (sigmoid(y)*2 - 0.5 + i)*640.0f/net_grid;
                w = pow(sigmoid(w)*2,2) * anchors[n*2];
                h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];

                double r_x = x - w/2;
                double r_y = y - h/2;
                Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                label_input.push_back(idx - 5);
            }

    std::cout <<  "o_rect.size(): " << o_rect.size() << std::endl;

    if(o_rect.size() == 0) return false;
    else return true;
}

bool Detector::parse_yolov5_2(const Blob::Ptr &blob, float cof_threshold, 
                    vector<Rect>& o_rect, vector<float>& o_rect_cof, vector<int>& label_input) {
    // 
    LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
    const float *output_blob = blobMapped.as<float *>();

    // int item_size = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    // int max_index = static_cast<int>(blob->getTensorDesc().getDims()[1]);
    int item_size = 7;
    int max_index = 25200;
    std::cout << "max_index: " << max_index << std::endl;
    std::cout << "item_size: " << item_size << std::endl;
    // 
    for (int i=0; i<max_index; i++) {
        double box_prob = output_blob[item_size * i + 4];  // 0,1,2,3 x,y,w,h;  4 conf;  5,6 class1,class2;
        box_prob = sigmoid(box_prob);
        //框置信度不满足则整体置信度不满足
        // 框内 [有物体] 的概率
        if(box_prob < cof_threshold)    
            continue;

        //注意此处输出为中心点坐标,需要转化为角点坐标
        double x = output_blob[item_size * i + 0];
        double y = output_blob[item_size * i + 1];
        double w = output_blob[item_size * i + 2];
        double h = output_blob[item_size * i + 3];

        // agxmax();
        double max_prob = 0;
        int idx=0;
        for(int t=5; t<item_size; ++t){
            double tp= output_blob[item_size * i + t];
            tp = sigmoid(tp);
            if(tp > max_prob){
                max_prob = tp;
                idx = t;
            }
        }

        // 框内物体类别置信度
        float cof = box_prob * max_prob;
        //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
        if(cof < cof_threshold)
            continue;

        double r_x = x - w/2;
        double r_y = y - h/2;
        Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
        o_rect.push_back(rect);
        o_rect_cof.push_back(cof);
        label_input.push_back(idx - 5);
    }
    std::cout <<  "o_rect.size(): " << o_rect.size() << std::endl;
    if(o_rect.size() == 0) return false;
    else return true;
}

//初始化
bool Detector::init(string xml_path,double cof_threshold,double nms_area_threshold){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    //输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    // _network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Detector::uninit(){
    return true;
}

//处理图像获取结果
bool Detector::process_frame(Mat& inframe,vector<Object>& detected_objects){
    if(inframe.empty()){
        cout << "无效图片输入" << endl;
        return false;
    }
    resize(inframe,inframe,Size(640,640));
    cvtColor(inframe,inframe,COLOR_BGR2RGB);
    size_t img_size = 640*640;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    //nchw
    for(size_t row =0;row<640;row++){
        for(size_t col=0;col<640;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*640 + col] = float(inframe.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }
    //执行预测
    auto start_ = chrono::high_resolution_clock::now();
    infer_request->Infer();
    auto end_ = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_ = end_ - start_;
    cout<<"infer: "<<diff_.count()<<" s" << endl;
    //获取各层结果
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    vector<int> label;
    int s[3] = {80, 40, 20};
    int i=0;

    auto start = chrono::high_resolution_clock::now();

    static int _i = 0;
    for (auto &output : _outputinfo) {
        // std::cout << std::endl;
        // std::cout << ++_i << std::endl;
        if (++_i == 4 ) {
            continue;
        }

        // std::cout << output.first << std::endl;

        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);

        // std::cout << "h:"<< static_cast<int>(blob->getTensorDesc().getDims()[2]) << std::endl;
        // std::cout << "w:"<< static_cast<int>(blob->getTensorDesc().getDims()[3]) << std::endl;
        // std::cout << "b:"<< static_cast<int>(blob->getTensorDesc().getDims()[0]) << std::endl; 
        // std::cout << "anchor:"<< static_cast<int>(blob->getTensorDesc().getDims()[1]) << std::endl;

        parse_yolov5(blob,s[i],_cof_threshold,origin_rect,origin_rect_cof,label);
        ++i;
    }
    
    // // auto output = _outputinfo["output"];
    // // auto output_name = output.first;
    // // Blob::Ptr blob = infer_request->GetBlob(output_name);
    // Blob::Ptr blob = infer_request->GetBlob("output");

    // // std::cout << output.first << std::endl;
    // std::cout << "h:"<< static_cast<int>(blob->getTensorDesc().getDims()[2]) << std::endl;
    // std::cout << "w:"<< static_cast<int>(blob->getTensorDesc().getDims()[3]) << std::endl;
    // std::cout << "b:"<< static_cast<int>(blob->getTensorDesc().getDims()[0]) << std::endl; 
    // std::cout << "anchor:"<< static_cast<int>(blob->getTensorDesc().getDims()[1]) << std::endl;

    // parse_yolov5_2(blob ,_cof_threshold, origin_rect, origin_rect_cof, label);
    

    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"output_test: "<<diff.count()<<" s" << endl;
    
    //后处理获得最终检测结果
    vector<int> final_id;

    dnn::NMSBoxes(origin_rect,origin_rect_cof,_cof_threshold,_nms_area_threshold,final_id);   

    //根据final_id获取最终结果
    for(int i=0;i<final_id.size();++i){
        Rect resize_rect= origin_rect[final_id[i]];
        detected_objects.push_back(Object{
            origin_rect_cof[final_id[i]],
            names2[label[final_id[i]]],
            resize_rect,
            label[final_id[i]],
            std::abs(inframe.cols * 0.5  - resize_rect.y),
            std::abs(inframe.rows * 0.5  - resize_rect.x),
            std::pow(inframe.rows * 0.8 -  resize_rect.y,2) + std::pow(inframe.cols * 0.5 - resize_rect.x,2)
        });
    }
    return true;
}

//以下为工具函数
double Detector::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

vector<int> Detector::get_anchors(int net_grid){
    vector<int> anchors(6);
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326};
    if(net_grid == 80){
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return anchors;
}


