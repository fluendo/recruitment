#include "ReIdEngine.hh"

bool ReIdEngine::BuildInferEngine(std::string& model_path, const std::shared_ptr<nvinfer1::IBuilder>& builder,
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

	auto profile = builder->createOptimizationProfile();
	if (!profile) {
		return false;
	}

	profile->setDimensions(infer_network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4(1, 112, 112, 3));
	profile->setDimensions(infer_network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4(1, 112, 112, 3));
	profile->setDimensions(infer_network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4(1, 112, 112, 3));

	infer_config->addOptimizationProfile(profile);
	infer_config->setFlag(BuilderFlag::kFP16);
	infer_config->setBuilderOptimizationLevel(5);
	std::cout << "Setting flag..." << "\n";
	// infer_config->setFlag(BuilderFlag::kFP16);
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

void ReIdEngine::InitInfer(Dimensions dims_in) {
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
	cudaStreamSynchronize(this->stream_);
}

void ReIdEngine::Infer(float* host_input, float* host_output) {
	int active_device;
	cudaGetDevice(&active_device);
	if (active_device != this->cuda_idx_) {
		cudaSetDevice(this->cuda_idx_);
	}
	cudaMemcpyAsync(this->infer_bindings_[this->infer_input_tensors_[0].index_], host_input,
					this->infer_input_tensors_[0].size_, cudaMemcpyHostToDevice, this->stream_);
	cudaStreamSynchronize(this->stream_);
	SetTensorAddr(this->infer_input_tensors_, this->infer_output_tensors_, this->infer_bindings_, this->context_);
	auto status = this->context_->enqueueV3(this->stream_);
	cudaStreamSynchronize(this->stream_);
	cudaMemcpyAsync(host_output, this->infer_bindings_[this->infer_output_tensors_[0].index_],
					this->infer_output_tensors_[0].size_, cudaMemcpyDeviceToHost, this->stream_);
	cudaStreamSynchronize(this->stream_);
	DecodeRes(host_output);
}

void ReIdEngine::DecodeRes(float* host_emb, int dim) {
	float norm = 0.f;
	for (int i = 0; i < dim; ++i) norm += host_emb[i] * host_emb[i];
	norm = std::sqrt(norm);
	for (int i = 0; i < dim; ++i) host_emb[i] /= norm;
}
