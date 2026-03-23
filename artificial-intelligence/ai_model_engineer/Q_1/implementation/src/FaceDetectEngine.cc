#include "FaceDetectEngine.hh"


/**
 * @brief Intersection over Union function
 * 
 * @param a Face 1 bounding box
 * @param b Face 2 bounding box
 * @return float 
 */
static float iou(const FaceSegm& a, const FaceSegm& b) {
	int ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
	int ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
	int bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
	int bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;

	int ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
	int ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);

	int inter = std::max(0, ix2 - ix1) * std::max(0, iy2 - iy1);
	if (inter == 0) return 0.f;

	int area_a = a.w * a.h;
	int area_b = b.w * b.h;
	return static_cast<float>(inter) / (area_a + area_b - inter);
}


/**
 * @brief Non-maximum suppression, removes duplicate detections
 * 
 * @param boxes Detected bounding boxes vector
 * @param iou_thresh Intersection over Union threshold
 */
static void nms(std::vector<FaceSegm>& boxes, float iou_thresh = 0.45f) {
	// Sort by confidence descending
	std::sort(boxes.begin(), boxes.end(), [](const FaceSegm& a, const FaceSegm& b) { return a.conf > b.conf; });

	std::vector<bool> suppressed(boxes.size(), false);
	std::vector<FaceSegm> result;

	for (size_t i = 0; i < boxes.size(); ++i) {
		if (suppressed[i]) continue;

		result.push_back(boxes[i]);

		for (size_t j = i + 1; j < boxes.size(); ++j) {
			if (suppressed[j]) continue;
			if (iou(boxes[i], boxes[j]) > iou_thresh) suppressed[j] = true;
		}
	}

	boxes = std::move(result);
}

bool FaceDetectEngine::BuildInferEngine(std::string& model_path, const std::shared_ptr<nvinfer1::IBuilder>& builder,
										const std::shared_ptr<nvinfer1::IRuntime>& runtime, cudaStream_t& profileStream,
										mLogger& logger) {
	std::cout << "CUDA device: " << this->cuda_idx_ << "\n";
	std::cout << "Creating network" << "\n";
	auto infer_network =
		builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
	std::cout << "Created network" << "\n";
	if (!infer_network) {
		std::cerr << "Failed to create infer network!" << "\n";
		return false;
	}
	auto parser = createParser(*infer_network, logger);
	std::string onnx_path = model_path + ".onnx";
	std::string engine_path = model_path + ".engine";
	std::cout << onnx_path << "\n";
	auto parsed = parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
	if (!parsed) {
		std::cerr << "Not parsed" << "\n";
		return false;
	} else {
		std::cout << "Parsed" << "\n";
	}

	for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
		std::cout << "Error : " << parser->getError(i)->desc() << "\n";
	}

	auto infer_config = builder->createBuilderConfig();
	if (!infer_config) {
		std::cout << "Infer config failed" << "\n";
		return false;
	}

	std::cout << "Setting flag..." << "\n";
	infer_config->setFlag(BuilderFlag::kFP16);
	infer_config->setBuilderOptimizationLevel(5);
	
	infer_config->setProfileStream(profileStream);
	std::shared_ptr<nvinfer1::IHostMemory> infer_plan{builder->buildSerializedNetwork(*infer_network, *infer_config)};
	if (!infer_plan) {
		std::cout << "Infer plan failed" << "\n";
		return false;
	}
	std::cout << "Deserializing engine!" << "\n";
	this->m_engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
		runtime->deserializeCudaEngine(infer_plan->data(), infer_plan->size()), InferDeleter());
	std::cout << "Deserialized successfully!" << "\n";
	if (!this->m_engine_) {
		std::cerr << "Infer engine failed" << "\n";
		return false;
	} else {
		std::cout << "Infer engine successful" << "\n";
		nvinfer1::IHostMemory* data = this->m_engine_->serialize();
		std::ofstream file(engine_path, std::ios::binary);
		if (!file) {
			return false;
		}
		file.write(reinterpret_cast<const char*>(data->data()), data->size());
		return true;
	}
}

void FaceDetectEngine::InitInfer(Dimensions dims_in) {
	std::cout << "Model path: " << this->model_path_ << "\n";
	int active_device;
	cudaGetDevice(&active_device);
	if (active_device != this->cuda_idx_) {
		cudaSetDevice(this->cuda_idx_);
	}
	mLogger logger;
	cudaStreamCreate(&this->stream_);

	dims_in_ = dims_in;
	// creating cuda runtime and engine builder
	this->m_runtime_ = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(logger));
	this->builder_ = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));

	// deserializing or building main engine
	if (DeserializeEngine(this->m_engine_, this->model_path_, logger)) {
		std::cout << "Deserialized engine successfully!" << "\n";
	} else {
		std::cout << "Parsing onnx..." << "\n";
		bool res = BuildInferEngine(this->model_path_, this->builder_, this->m_runtime_, this->stream_, logger);
		if (res) {
			std::cout << "Parsed onnx successfully!" << "\n";
		} else {
			std::cout << "Parsed onnx failed!" << "\n";
		}
	}

	std::cout << "Creating main context\n";
	this->context_ = std::shared_ptr<nvinfer1::IExecutionContext>(this->m_engine_->createExecutionContext());
	GetTensorBindings(this->infer_input_tensors_, this->infer_output_tensors_, this->m_engine_, this->context_);

	std::cout << "Get main bindings\n";
	this->infer_bindings_ = std::vector<void*>(this->infer_input_tensors_.size() + this->infer_output_tensors_.size());

	// allocating main tensors
	for (auto& input : this->infer_input_tensors_) {
		CudaMallocTensor(this->infer_bindings_, input.index_, input.size_);
	}
	for (auto& output : this->infer_output_tensors_) {
		CudaMallocTensor(this->infer_bindings_, output.index_, output.size_);
	}

	src_bytes = dims_in_.height * dims_in_.width * dims_in_.channels * sizeof(uint8_t);
	cudaMalloc(&device_src_img_, src_bytes);
	cudaStreamSynchronize(this->stream_);
}

void FaceDetectEngine::Infer(uint8_t* host_input, float* host_output) {
	int active_device;
	cudaGetDevice(&active_device);
	if (active_device != this->cuda_idx_) {
		cudaSetDevice(this->cuda_idx_);
	}
	cudaMemcpyAsync(device_src_img_, host_input, this->infer_input_tensors_[0].size_ / sizeof(float),
					cudaMemcpyHostToDevice, this->stream_);
	cudaStreamSynchronize(this->stream_);

	launch_preprocess(device_src_img_, (float*)this->infer_bindings_[this->infer_input_tensors_[0].index_],
					  dims_in_.height, dims_in_.width, this->stream_);
	cudaStreamSynchronize(this->stream_);

	SetTensorAddr(this->infer_input_tensors_, this->infer_output_tensors_, this->infer_bindings_, this->context_);
	auto status = this->context_->enqueueV3(this->stream_);
	cudaStreamSynchronize(this->stream_);
	cudaMemcpyAsync(host_output, this->infer_bindings_[this->infer_output_tensors_[0].index_],
					this->infer_output_tensors_[0].size_, cudaMemcpyDeviceToHost, this->stream_);
}

void FaceDetectEngine::DecodeRes(float* host_ptr, std::vector<FaceSegm>& bound_boxes, int orig_w, int orig_h,
								 float thresh) {
	const int NUM_PROPOSALS = 8400;
	const float scale = std::max(orig_w, orig_h) / 640.0f;	// 1920/640 = 3.0

	bound_boxes.clear();

	for (int j = 0; j < NUM_PROPOSALS; ++j) {
		float conf = host_ptr[4 * NUM_PROPOSALS + j];
		if (conf < thresh) continue;

		float cx_f = host_ptr[0 * NUM_PROPOSALS + j];
		float cy_f = host_ptr[1 * NUM_PROPOSALS + j];
		float w_f = host_ptr[2 * NUM_PROPOSALS + j];
		float h_f = host_ptr[3 * NUM_PROPOSALS + j];

		float x1 = (cx_f - w_f * 0.5f) * scale;
		float y1 = (cy_f - h_f * 0.5f) * scale;
		float x2 = (cx_f + w_f * 0.5f) * scale;
		float y2 = (cy_f + h_f * 0.5f) * scale;

		x1 = std::clamp(x1, 0.f, (float)orig_w);
		y1 = std::clamp(y1, 0.f, (float)orig_h);
		x2 = std::clamp(x2, 0.f, (float)orig_w);
		y2 = std::clamp(y2, 0.f, (float)orig_h);

		FaceSegm fs;
		fs.cx = static_cast<int>((x1 + x2) * 0.5f);
		fs.cy = static_cast<int>((y1 + y2) * 0.5f);
		fs.w = static_cast<int>(x2 - x1);
		fs.h = static_cast<int>(y2 - y1);
		fs.conf = conf;

		bound_boxes.push_back(fs);
	}

	nms(bound_boxes);
}
