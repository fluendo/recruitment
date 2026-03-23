#pragma once
#include "Types.hh"
#include "InferEngine.hh"
#include "preproc.hh"

/**
 * @brief Derived class for face detection model
 */
class FaceDetectEngine : public InferEngine {
   public:
	/**
	 * @brief Default constructor
	 *
	 */
	FaceDetectEngine() {
		this->model_path_ = "";
		this->cuda_idx_ = 0;
	}

	/**
	 * @brief Parametrized constructor
	 *
	 *
	 * @param model_path
	 * @param cuda_idx
	 */
	FaceDetectEngine(std::string model_path, int cuda_idx) {
		this->model_path_ = model_path;
		this->cuda_idx_ = cuda_idx;
	}

	/**
	 * @brief Initializing TensorRT inference
	 *
	 * @param dims_in dimensions of input image
	 * @param dims_out dimensions of output result
	 * @param delta size of top and bottom border in pixels (4 for 1920x1080)
	 */
	void InitInfer(Dimensions dims_in);

	/**
	 * @brief Perform inference
	 * 
	 * @param host_input pointer to BGR frame
	 * @param host_output bounding box data host pointer
	 */
	void Infer(uint8_t* host_input, float* host_output);

	/**
	 * @brief 
	 * 
	 * @param host_ptr host pointer input
	 * @param bound_boxes bounding boxes result scaled to original input image size
	 * @param orig_w original img width
	 * @param orig_h original img heught
	 * @param thresh confidence threshold
	 */
	void DecodeRes(float* host_ptr, std::vector<FaceSegm>& bound_boxes, int orig_w, int orig_h, float thresh);

   private:
	/**
	 * @brief Build primary model engine from onnx
	 *
	 * @param model_path absolute path to model
	 * @param builder reference to builder object
	 * @param runtime reference to runtime object
	 * @param profileStream reference to cuda profile stream
	 * @param logger reference to logger instance
	 * @return true
	 * @return false
	 */
	bool BuildInferEngine(std::string& model_path, const std::shared_ptr<nvinfer1::IBuilder>& builder,
						  const std::shared_ptr<nvinfer1::IRuntime>& runtime, cudaStream_t& profileStream,
						  mLogger& logger);

	Dimensions dims_in_; /**< Input image dimensions {width, height, channels} */

	uint8_t* device_src_img_; /**< Device image pointer for pre-processing purposes */

	size_t src_bytes; /**< Size of source image in bytes*/
};
