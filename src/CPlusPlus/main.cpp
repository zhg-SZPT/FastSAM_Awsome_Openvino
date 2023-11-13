#include <iostream>
#include <memory>
#include <string>

#include <openvino/openvino.hpp>

#include <opencv2/opencv.hpp>

#include "slog.hpp"

#include "FastSAM.h"


void printInputAndOutputsInfo(const ov::Model& network) {
    slog::info<< "model name: " << network.get_friendly_name() << slog::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        slog::info<< "    inputs" << slog::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        slog::info<< "        input name: " << name << slog::endl;

        const ov::element::Type type = input.get_element_type();
        slog::info<< "        input type: " << type << slog::endl;

        const ov::Shape shape = input.get_shape();
        slog::info<< "        input shape: " << shape << slog::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        slog::info<< "    outputs" << slog::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        slog::info<< "        output name: " << name << slog::endl;

        const ov::element::Type type = output.get_element_type();
        slog::info<< "        output type: " << type << slog::endl;

        const ov::Shape shape = output.get_shape();
        slog::info<< "        output shape: " << shape << slog::endl;
    }
}








int main(int argc, char* argv[])
{
    if(argc != 3) {
        slog::info << "Usage:" << argv[0] << " <xml_model_path> <infer_image_path>\n";
        return -1;
    }

    std::string xml_path = argv[1];
    std::string image_path = argv[2];

    FastSAM fastsam;
    if(fastsam.Initialize(xml_path, 0.3, 0.2, true)) {
        fastsam.Infer(image_path);
    }

    

    slog::info << "Fastsam deploy with openvino!\n";

    return 0;
}