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
    
    /// @brief 初始化模型ir 
    /// @param xml_path 指向xml模型网络结构文件
    /// @param conf 置信度阈值
    /// @param iou 交并比阈值
    /// @param useGpu 是否使用gpu
    /// @return 
    bool Initialize(const std::string& xml_path, float conf, float iou, bool useGpu);

    /// @brief 推理
    /// @param image_path 图像路径 
    /// @return 
    void Infer(const std::string& image_path);

    /// @brief 推理
    /// @param image 
    /// @return 
    cv::Mat Infer(const cv::Mat& image);

private:
    /// @brief 后处理，主要是非极大抑制 + 还原坐标 + 掩码处理
    /// @param preds 输出的两个维度
    /// @param oriImage 原始图像
    /// @return 返回处理好的结果
    std::vector<cv::Mat> Postprocess(const cv::Mat& oriImage);

    /// @brief 
    /// @return 
    cv::Mat BuildOutput0();

    /// @brief 
    /// @return 
    cv::Mat BuildOutput1();
    
    /// @brief bbox在原始图像中的坐标位置
    /// @param box 
    /// @param oriImage 
    void ScaleBoxes(cv::Mat& box, const cv::Size& oriSize);

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
    std::vector<cv::Mat> NMS(cv::Mat& prediction, int max_det = 300);

    /// @brief 将中心坐标+宽高 --> 左上角坐标+右下加坐标
    /// @param box 
    void xywh2xyxy(cv::Mat &box);


private:
    /// @brief 将得到的掩码结果绘制到这个图像上， 需要原图
    /// @param image 原始图像
    /// @param vremat 掩码矩阵
    cv::Mat Render(const cv::Mat& image, const std::vector<cv::Mat>& vremat);


private:

    /// @brief 预处理，将图像resize,并且转换排列顺序
    /// @param image 预处理的图像
    /// @return 返回处理好的tensor 
    ov::Tensor Preprocess(cv::Mat& image);

    /// @brief resize 图像的大小以符合模型输入的大小
    /// @param image 需要处理的图像
    /// @return 返回是否转换成功
    bool ConvertSize(cv::Mat& image);


    /// @brief 将图像由bgr转化为rgb、并且排列顺序由NHWC转化为NCHW，将u8->f32,最后将转化好的所有值归一化到[0-1]保存到input_data
    /// @param image  
    /// @return 
    bool ConvertLayout(cv::Mat& image);

    /// @brief 根据这个input_data构造这个ov::tensor
    /// @return 
    ov::Tensor BuildTensor();


private:
    /// @brief 从读取到的模型中获取输入输出的shape 
    /// @return 是否成功
    bool ParseArgs();

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
    

    float m_conf;       // 置信度阈值
    float m_iou;        // 交并比阈值

    std::vector<float> input_data;

    int input_width = 0;            // 模型输入的宽度 
    int input_height = 0;           // 模型输入的高度
    int input_channel = 3;
    ov::Shape model_input_shape;    // 模型的输入形状
    ov::Shape model_output0_shape;  // 模型输出第一个维度的形状
    ov::Shape model_output1_shape;  // 模型输出第二个维度的形状


    float ratio = 1.0f;     // 缩放比例, 保留用于输出后还原输出的坐标
    float dw = 0.f;         //
    float dh = 0.f;

    int mw = 160;           // 输出掩膜的宽度
    int mh = 160;           // 输出的掩膜的高度

    cv::Mat m_image;
};
