#include <iostream>
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include <opencv2/opencv.hpp>

#include <chrono>

#include "slog.hpp"

#include "FastSAM.h"



int main(int argc, char* argv[])
{
    if(argc != 3) {
        slog::info << "Usage:" << argv[0] << " <xml_model_path> <infer_image_path>\n";
        return -1;
    }

    std::string xml_path = argv[1];
    std::string image_path = argv[2];

    FastSAM fastsam;
    if(fastsam.Initialize(xml_path, 0.4, 0.3, true)) {
        
        auto start = std::chrono::steady_clock::now();
        fastsam.Infer(image_path);
        auto end = std::chrono::steady_clock::now();
        

        auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();;
        slog::info << "infer time:" << tt << " ms. \n";
    }

    
        

    slog::info << "Fastsam deploy with openvino!\n";

    return 0;
}
