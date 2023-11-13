#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include <array>

class FastSAM
{
public:
    FastSAM() {};
    ~FastSAM(){};
    
    /// @brief Initialize the fastsam model 
    /// @param xml_path the model xml config path
    /// @param conf 
    /// @param iou 
    /// @param useGpu 
    /// @return 
    bool Initialize(const std::string& xml_path, float conf, float iou, bool useGpu);

    /// @brief 
    /// @param image_path 
    /// @return 
    void Infer(const std::string& image_path);


private:

    /// @brief 推理
    /// @param image 
    /// @return 
    cv::Mat Infer(const cv::Mat& image);

    /// @brief 从读取到的模型中获取输入输出的shape 
    /// @return 是否成功
    bool ParseArgs();

    
    /// @brief bbox在原始图像中的坐标位置
    /// @param box 
    /// @param oriImage 
    void ScaleBoxes(cv::Mat& box, const cv::Mat& oriImage);

    /// @brief 根据输入的图像、原型、掩码等信息，生成并返回一组处理后的掩码图像及其对应的边界框信息。掩码图像是通过矩阵运算、变换和处理得到的，最终被存储在向量中并返回。
    /// @param oriImage 原始输入图像
    /// @param protos 模型输出的第二个维度
    /// @param masks_in 掩码
    /// @param bboxes 坐标
    /// @param shape 生成掩码图像的大小
    /// @return 
    std::vector<cv::Mat> ProcessMaskNative(const cv::Mat& oriImage, cv::Mat& protos, cv::Mat& masks_in, cv::Mat& bboxes, cv::Size shape);

    /// @brief 非极大抑制，解析原始数据并且 过滤掉置信度过低的、和iou较高的
    /// @param vreMat 保存结果bbox 和 mask
    /// @param prediction 模型输出的第一个维度
    /// @param max_det 检测的最大数量值
    void NMS(std::vector<cv::Mat>& vreMat, cv::Mat& prediction, int max_det = 300);

    /// @brief 将中心坐标+宽高 --> 左上角坐标+右下加坐标
    /// @param box 
    void xywh2xyxy(cv::Mat &box);

    /// @brief 将得到的掩码结果绘制到这个图像上， 需要原图
    /// @param image 原始图像
    /// @param vremat 掩码矩阵
    void Render(cv::Mat& image, const std::vector<cv::Mat>& vremat);

    /// @brief 转换颜色编码BGR -> RGB, 将存储顺序 BWHC -> BCWH
    /// @param image 
    void Normalize2Vec(cv::Mat& image);
    
    /// @brief resize the image, and create input tensor
    /// @param image 
    /// @return 
    ov::Tensor Preprocess(const cv::Mat& image);


    /// @brief 后处理，主要是非极大抑制 + 还原坐标 + 掩码处理
    /// @param preds 输出的两个维度
    /// @param oriImage 原始图像
    /// @return 返回处理好的结果
    std::vector<cv::Mat> Postprocess(std::vector<cv::Mat> &preds, const cv::Mat& oriImage);


    /// @brief 构建PrePostProcessor
    /// @return 是否成功
    bool BuildProcessor();

    /// @brief 判断是否可以使用GPU
    /// @param core 
    /// @return 返回true 如果可以使用GPU
    bool IsGpuAvaliable(const ov::Core& core);

private:
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;

    ov::Core m_core;
    ov::InferRequest m_request;
    std::shared_ptr<ov::preprocess::PrePostProcessor> m_ppp;
    

    float m_conf;
    float m_iou;

    std::vector<float> input_data;

    int input_width = 0;
    int input_height = 0;
    ov::Shape model_input_shape;
    ov::Shape model_output0_shape;
    ov::Shape model_output1_shape;

    float ratio = 1.0f;
    float dw = 0.f;
    float dh = 0.f;

    int mw = 160;
    int mh = 160;

    cv::Mat m_image;
    //PaddingInfo m_padding = {0, 0, 0, 0};
};