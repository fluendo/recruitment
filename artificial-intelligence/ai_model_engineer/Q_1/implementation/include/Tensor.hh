#pragma once
#include <string>

#include "NvInfer.h"
#include "NvOnnxParser.h"

using namespace nvinfer1;

/**
 * @brief Tensor info class
 */
class Tensor {
   public:
	/**
	 * @brief Default constructor for Tensor object
	 */
	Tensor() : name_(""), index_(0), size_(0), ptr_(nullptr) {}

	/**
	 * @brief Parametrized constructor for Tensor object
	 */
	Tensor(std::string& name, int index, Dims dimensions, size_t size)
		: name_(name), index_(index), size_(size), ptr_(new void*[size]) {
		this->dimensions_ = dimensions;
	}

	/**
	 * @brief Parametrized constructor for Tensor object
	 */
	Tensor(std::string& name, int index, Dims dimensions) : name_(name), index_(index), size_(0), ptr_(nullptr) {
		this->dimensions_ = dimensions;
	}

	// ~Tensor() {
	// 	if (ptr_) {
	// 		delete[] static_cast<void**>(ptr_);
	// 		ptr_ = nullptr; // Safety: avoid dangling pointers
	// 	}
	// }

	size_t size_;	   /**< Tensor size in bytes */
	std::string name_; /**< Tensor name in model */
	int index_;		   /**< Tensor index in model */
	Dims dimensions_;  /**< Tensor dimensions */
	void* ptr_;		   /**< Tensor data pointer*/
};
