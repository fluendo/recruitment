#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <thread>
#include <queue>
#include <chrono>
#include <sstream>
#include <fstream>
#include <numeric>
#include <functional>
#include <tuple>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "Tensor.hh"
#include "Types.hh"
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using namespace nvinfer1;
using namespace nvonnxparser;

/**
 * @brief TRT infer object deleter class
 */
struct InferDeleter {
	template <typename T>
	void operator()(T* obj) const {
		delete obj;
	}
};

/**
 * @brief TensorRT logger class
 */
class mLogger : public ILogger {
   public:
	mLogger() {}

	void log(Severity severity, const char* msg) noexcept override {
		// suppress info-level messages
		if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
	}
};

/**
 * @brief Base class for TensorRT model
 */
class InferEngine {
   public:
	/**
	 * @brief Destroy inference and deallocate resources
	 */
	~InferEngine();

   protected:
	std::vector<void*> infer_bindings_; /**< Vector of pointers model bindings */

	std::vector<Tensor> infer_input_tensors_; /**< Vector of inference model input tensors */

	std::vector<Tensor> infer_output_tensors_; /**< Vector of inference model output tensors */

	std::string model_path_; /**< Path to model file */

	std::shared_ptr<nvinfer1::IExecutionContext> context_; /**< Pointer to inference model execturion context object */

	std::shared_ptr<nvinfer1::ICudaEngine> m_engine_; /**< Pointer to inference model engine object */

	int cuda_idx_; /**< Cuda device index for model */

	cudaStream_t stream_; /**< Cuda stream object */

	std::shared_ptr<nvinfer1::IBuilder> builder_; /**< Pointer to model builder object */

	std::shared_ptr<nvinfer1::IRuntime> m_runtime_; /**< Pointer to runtime object */

	/**
	 * @brief Get Tensor dimensions and i/o tensor names overload
	 *
	 * @param input_tensors reference to vector of input tensors
	 * @param output_tensors reference to vector of output tensors
	 * @param engine reference to engine pointer
	 * @param context reference to context_ pointer
	 */
	void GetTensorBindings(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
						   std::shared_ptr<nvinfer1::ICudaEngine>& engine,
						   std::shared_ptr<nvinfer1::IExecutionContext>& context);

	/**
	 * @brief Allocate cuda memory for tensor
	 *
	 * @param bindings reference to bindings vector
	 * @param idx reference to binding index
	 * @param size reference to binding size
	 */
	void CudaMallocTensor(std::vector<void*>& bindings, int idx, size_t size);

	/**
	 * @brief Set tensor address in memory
	 * @param input_tensors reference to input tensor vector
	 * @param output_tensors reference to output tensor vector
	 * @param bindings reference to network bindings
	 * @param context reference to context object
	 */
	void SetTensorAddr(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
					   std::vector<void*>& bindings, std::shared_ptr<nvinfer1::IExecutionContext>& context);

	/**
	 * @brief Get size of trt datatype
	 *
	 * @param datatype tensorrt datatype
	 * @return const int32_t size of data
	 */
	const int32_t GetSizeofDatatype(nvinfer1::DataType type);

	/**
	 * @brief Deserialize .engine file
	 *
	 * @param m_engine reference to engine object
	 * @param model_path absolute path to model
	 * @param logger reference to logger instance
	 * @return true
	 * @return false
	 */
	bool DeserializeEngine(std::shared_ptr<nvinfer1::ICudaEngine>& m_engine, std::string& model_path, mLogger& logger);

	/**
	 * @brief Get size of Tensor in bytes
	 *
	 * @param dims reference to tensor dimensions
	 * @param elem_size element size in bytes
	 * @return size_t
	 */
	size_t GetMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size);
};
