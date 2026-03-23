#pragma once
#include "Types.hh"
#include "InferEngine.hh"
#include <cmath>

/**
 * @brief Derived class for Reidentification model
 */
class ReIdEngine : public InferEngine {
   public:
	/**
	 * @brief Default constructor
	 *
	 */
	ReIdEngine() {
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
	ReIdEngine(std::string model_path, int cuda_idx) {
		this->model_path_ = model_path;
		this->cuda_idx_ = cuda_idx;
	}

	/**
	 * @brief Initializing TensorRT inference
	 *
	 * @param dims_in dimensions of input image
	 */
	void InitInfer(Dimensions dims_in);

	/**
	 * @brief Perform inference
	 * 
	 * @param host_input input RGB ROI 112x112 host pointer
	 * @param host_output output embedding vector host pointer
	 */
	void Infer(float* host_input, float* host_output);

	/**
	 * @brief Decode embedding results
	 * 
	 * @param host_emb embedding vector host pointer
	 * @param dim embedding vector size in bytes, default 512
	 */
	void DecodeRes(float* host_emb, int dim = 512);

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
};
