#include "FastSAM.h"
#include <algorithm>
#include <filesystem>
#include <algorithm>

#include "slog.hpp"


cv::Scalar RandomColor()
{
    int b = rand() % 256;
    int g = rand() % 256;
    int r = rand() % 256;
    return cv::Scalar(b, g, r);
}



bool FastSAM::Initialize(const std::string &xml_path, float conf, float iou,  bool useGpu)
{   
    m_conf = conf;
    m_iou = iou;

    if(!std::filesystem::exists(xml_path))
        return false;
    
    m_model = m_core.read_model(xml_path);

    if(!ParseArgs())
        return false;


    if(!BuildProcessor()) 
        return false;
    
    if(useGpu)
        m_compiled_model = m_core.compile_model(m_model, 
                IsGpuAvaliable(m_core) ? "GPU":"CPU");

    m_request = m_compiled_model.create_infer_request();

    return true;
}

void FastSAM::Infer(const std::string &image_path)
{
    try
    {
        cv::Mat image = cv::imread(image_path);
    
        cv::Mat rendered = Infer(image);

        std::string savedir = std::filesystem::current_path().string() + "/results";
        if(!std::filesystem::exists(savedir))
            std::filesystem::create_directory(savedir);
            
        std::string savepath = savedir + "/" + std::filesystem::path(image_path).filename().string();
        cv::imwrite(savepath, rendered);

        slog::info << "result save in:" << savepath << "\n";
    }
    catch(const std::exception& e)
    {
        slog::info << "Failed to Infer! ec: " <<e.what() << '\n';
    }
}

cv::Mat FastSAM::Infer(const cv::Mat &image)
{
    
    ov::Tensor input_tensor = Preprocess(image);

    m_request.set_input_tensor(input_tensor);
    m_request.infer();
   
    auto* p0 = m_request.get_output_tensor(0).data();
    auto* p1 = m_request.get_output_tensor(1).data();

    cv::Mat output0 = cv::Mat(model_output0_shape[1], model_output0_shape[2], CV_32F, p0);
    cv::Mat output1 = cv::Mat(model_output1_shape[1], model_output1_shape[2] * model_output1_shape[3], CV_32F, p1);
    
    std::vector<cv::Mat> preds = {output0, output1};
    std::vector<cv::Mat> result =  Postprocess(preds, image);

    cv::Mat renderImage = image.clone();
    Render(renderImage, result);

    return renderImage;
}

bool FastSAM::ParseArgs()
{
    try
    {
        model_input_shape = m_model->input().get_shape();
        model_output0_shape = m_model->output(0).get_shape();
        model_output1_shape = m_model->output(1).get_shape();
        
        slog::info  << "xml input shape:" << model_input_shape << "\n";
        slog::info << "xml output shape 0:" << model_output0_shape << "\n";
        slog::info << "xml output shape 1:" << model_output1_shape << "\n";

        input_height = model_input_shape[2];
        input_width = model_input_shape[3];

        slog::info << "model input height:" << input_height << " input width:" << input_width << "\n";

        mh = model_output1_shape[2];
        mw = model_output1_shape[3];

        slog::info << "model output mh:" << mh << " output mw:" << mw << "\n";
    }
    catch(const std::exception& e)
    {
        slog::info << "Failed to Parse Args. "<< e.what() << '\n';
        return false;
    }
    
    
    return true;
}

void FastSAM::ScaleBoxes(cv::Mat &box, const cv::Mat& oriImage)
{
    float *pxvec = box.ptr<float>(0);
    for (int i = 0; i < box.rows; i++) {
        pxvec = box.ptr<float>(i);
        pxvec[0] -= this->dw;   
        pxvec[0] = std::clamp(pxvec[0] * this->ratio, 0.f, (float)oriImage.cols);
        pxvec[1] -= this->dh;
        pxvec[1] = std::clamp(pxvec[1] * this->ratio, 0.f, (float)oriImage.rows);
        pxvec[2] -= this->dw;
        pxvec[2] = std::clamp(pxvec[2] * this->ratio, 0.f, (float)oriImage.cols);
        pxvec[3] -= this->dh;
        pxvec[3] = std::clamp(pxvec[3] * this->ratio, 0.f, (float)oriImage.rows);
    }
}

std::vector<cv::Mat> FastSAM::ProcessMaskNative(const cv::Mat &image, cv::Mat &protos, cv::Mat &masks_in, cv::Mat &bboxes, cv::Size shape)
{
    std::vector<cv::Mat> result;
    result.push_back(bboxes);

    cv::Mat matmulRes = (masks_in * protos).t(); // 矩阵相乘后转

    cv::Mat maskMat = matmulRes.reshape(bboxes.rows, {mh, mw});  // shape [bboxes.rows, 160, 160]  

    std::vector<cv::Mat> maskChannels;
    cv::split(maskMat, maskChannels);
    float target_size = input_height;
    int scale_dw = this->dw / target_size * mw;
    int scale_dh = this->dh / target_size * mh;
    cv::Rect roi(scale_dw, scale_dh, mw - 2 * scale_dw, mh - 2 * scale_dh);
    float *pxvec = bboxes.ptr<float>(0);
    for (int i = 0; i < bboxes.rows; i++) {
        pxvec = bboxes.ptr<float>(i);
        cv::Mat dest, mask;
        cv::exp(-maskChannels[i], dest);
        dest = 1.0 / (1.0 + dest);
        dest = dest(roi);
        cv::resize(dest, mask, image.size(), cv::INTER_LINEAR);
        cv::Rect roi(pxvec[0], pxvec[1], pxvec[2] - pxvec[0], pxvec[3] - pxvec[1]);
        cv::Mat temmask = mask(roi);
        cv::Mat boxMask = cv::Mat(image.size(), mask.type(), cv::Scalar(0.0));
        float rx = std::max(pxvec[0], 0.0f);
        float ry = std::max(pxvec[1], 0.0f);
        for (int y = ry, my = 0; my < temmask.rows; y++, my++) {
            float *ptemmask = temmask.ptr<float>(my);
            float *pboxmask = boxMask.ptr<float>(y);
            for (int x = rx, mx = 0; mx < temmask.cols; x++, mx++) {
                pboxmask[x] = ptemmask[mx] > 0.5 ? 1.0 : 0.0;
            }
        }
        result.push_back(boxMask);
    }

    return result;
}

void FastSAM::NMS(std::vector<cv::Mat> &vreMat, cv::Mat &prediction, int max_det)
{
    cv::Mat temData = cv::Mat();
    prediction = prediction.t(); // [37, 8400] --> [rows:8400, cols:37]
    float *pxvec = prediction.ptr<float>(0);
    
    for (int i = 0; i < prediction.rows; i++) {
        pxvec = prediction.ptr<float>(i);
        if (pxvec[4] > m_conf) {
            temData.push_back(prediction.rowRange(i, i + 1).clone());
        }
    }

    if (temData.rows == 0) {
        return;
    }

    cv::Mat box = temData.colRange(0, 4).clone();   // 取所有box的列的值
    cv::Mat cls = temData.colRange(4, 5).clone();   // 取所有类别得分值 
    cv::Mat mask = temData.colRange(5, temData.cols).clone(); // 取后面掩膜矩阵

    cv::Mat j = cv::Mat::zeros(cls.size(), CV_32F);
    cv::Mat dst;
    cv::hconcat(box, cls, dst); // 纵向concat
    cv::hconcat(dst, j, dst);
    cv::hconcat(dst, mask, dst); // dst = [box class j mask]  把这几个按照列concat 起来

    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    pxvec = dst.ptr<float>(0);
    for (int i = 0; i < dst.rows; i++) {
        pxvec = dst.ptr<float>(i);
        boxes.push_back(cv::Rect(pxvec[0], pxvec[1], pxvec[2], pxvec[3]));
        scores.push_back(pxvec[4]);
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, m_conf, m_iou, indices); 
    cv::Mat reMat;
    for (int i = 0; i < indices.size() && i < max_det; i++) {
        int index = indices[i];
        reMat.push_back(dst.rowRange(index, index + 1).clone());
    }
    box = reMat.colRange(0, 6).clone();
    xywh2xyxy(box);
    mask = reMat.colRange(6, reMat.cols).clone();

    vreMat.push_back(box);
    vreMat.push_back(mask);

    slog::info << "mask size:" << mask.rows << "\n";
}

void FastSAM::xywh2xyxy(cv::Mat &box)
{
    float *pxvec = box.ptr<float>(0);
    for (int i = 0; i < box.rows; i++) {
        pxvec = box.ptr<float>(i);
        float w = pxvec[2];
        float h = pxvec[3];
        float cx = pxvec[0];
        float cy = pxvec[1];
        pxvec[0] = cx - w / 2;
        pxvec[1] = cy - h / 2;
        pxvec[2] = cx + w / 2;
        pxvec[3] = cy + h / 2;
  }
}

void FastSAM::Render(cv::Mat &image, const std::vector<cv::Mat>& vremat)
{

    cv::Mat bbox = vremat[0];
    float *pxvec = bbox.ptr<float>(0);
    
    for (int i = 1; i < vremat.size(); i++) {
        cv::Mat mask = vremat[i];
        auto color = RandomColor();

        for (int y = 0; y < mask.rows; y++) {
        const float *mp = mask.ptr<float>(y);
        uchar *p = image.ptr<uchar>(y);
        for (int x = 0; x < mask.cols; x++) {
            if (mp[x] == 1.0) {
            p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
            p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
            p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
            }
            p += 3;
        }
        }
    }

}

void FastSAM::Normalize2Vec(cv::Mat &image)
{
    int row = image.rows;
    int col = image.cols;
    this->input_data.resize(row * col * image.channels());
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                float pix = image.ptr<unsigned char>(i)[j * 3 + c];
                // 将像素值归一化到[0, 1]范围
                this->input_data[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

ov::Tensor FastSAM::Preprocess(const cv::Mat &image)
{
    float height = (float)image.rows;
    float width = (float)image.cols;

    int target_size = input_height;
    float r = std::min(target_size / height, target_size / width);
    int padw = (int)std::round(width * r);
    int padh = (int)std::round(height * r);

    
    if((int)width != padw || (int)height != padh) 
        cv::resize(image, m_image, cv::Size(padw, padh));
    else 
        m_image = image.clone();

    float _dw = target_size - padw;
    float _dh = target_size - padh;
    _dw /= 2.0f;
    _dh /= 2.0f;
    int top = int(std::round(_dh - 0.1f));
    int bottom = int(std::round(_dh + 0.1f));
    int left = int(std::round(_dw - 0.1f));
    int right = int(std::round(_dw + 0.1f));
    cv::copyMakeBorder(m_image, m_image, top, bottom, left, right, cv::BORDER_CONSTANT,
                        cv::Scalar(114, 114, 114));

    this->ratio = 1 / r;
    this->dw = _dw;
    this->dh = _dh;

    Normalize2Vec(m_image);

    return ov::Tensor(ov::element::f32, ov::Shape({1, 3, (unsigned long)input_height, (unsigned long)input_width}), input_data.data());    
}


std::vector<cv::Mat> FastSAM::Postprocess(std::vector<cv::Mat> &preds, const cv::Mat& oriImage)
{
    std::vector<cv::Mat> result;

    std::vector<cv::Mat> remat;
    NMS(remat, preds[0], 100);
    cv::Mat proto = preds[1];
    cv::Mat box = remat[0];
    cv::Mat mask = remat[1];
    ScaleBoxes(box, oriImage);

    return ProcessMaskNative(oriImage, proto, mask, box, oriImage.size());
}

bool FastSAM::BuildProcessor()
{
    try
    {
        m_ppp = std::make_shared<ov::preprocess::PrePostProcessor>(m_model);

        m_ppp->input().tensor()
            .set_shape(ov::Shape({1, 3, 640, 640}))
            .set_element_type(ov::element::f32) 
            .set_color_format(ov::preprocess::ColorFormat::RGB)
            .set_layout("NCHW");

        // m_ppp->input().preprocess()
        //     .convert_layout("NCHW");

        
        m_model = m_ppp->build();
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to build the model processor!\n" << e.what() << '\n';
        return false;
    }

    slog::info << "build successfully!\n";
    return true;
}

bool FastSAM::IsGpuAvaliable(const ov::Core& core)
{
    std::vector<std::string> avaliableDevice = core.get_available_devices();
    
    auto iter = std::find(avaliableDevice.begin(), avaliableDevice.end(), "GPU");

    return iter != avaliableDevice.end();
}
