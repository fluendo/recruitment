#include "InferEngine.hh"

const int32_t InferEngine::GetSizeofDatatype(nvinfer1::DataType datatype) {
	switch (datatype) {
		case nvinfer1::DataType::kFLOAT:
			return sizeof(float);
		case nvinfer1::DataType::kHALF:
			return sizeof(__half);
		case nvinfer1::DataType::kINT8:
			return sizeof(int8_t);
		case nvinfer1::DataType::kINT32:
			return sizeof(uint32_t);
		case nvinfer1::DataType::kBOOL:
			return sizeof(bool);
		case nvinfer1::DataType::kUINT8:
			return sizeof(uint8_t);
		default:
			return 0;
	}
}

InferEngine::~InferEngine() {
	this->context_.reset();

	this->m_engine_.reset();
	this->m_runtime_.reset();
	cudaFree(this->infer_bindings_.data());
	cudaStreamDestroy(this->stream_);
	this->stream_ = nullptr;
}

void InferEngine::GetTensorBindings(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
									std::shared_ptr<nvinfer1::ICudaEngine>& engine,
									std::shared_ptr<nvinfer1::IExecutionContext>& context) {
	std::ostringstream bindings_info;
	bindings_info << "Model bindings:\n";

	int nb_io_tensors = engine->getNbIOTensors();

	for (int i = 0; i < nb_io_tensors; i++) {
		const char* tensor_name = engine->getIOTensorName(i);
		std::string name(tensor_name);

		nvinfer1::Dims dims = context->getTensorShape(tensor_name);

		bool is_dynamic = false;
		for (int d = 0; d < dims.nbDims; d++) {
			if (dims.d[d] == -1) {
				is_dynamic = true;
				break;
			}
		}

		size_t size = 0;
		if (!is_dynamic) {
			size = GetMemorySize(dims, GetSizeofDatatype(engine->getTensorDataType(tensor_name)));
		}

		bindings_info << (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT ? "  Input  "
																								 : "  Output  ");
		bindings_info << i << ": " << tensor_name;
		for (int d = 0; d < dims.nbDims; d++) {
			bindings_info << (d == 0 ? ", " : "x") << dims.d[d];
		}
		bindings_info << ", type size: " << GetSizeofDatatype(engine->getTensorDataType(tensor_name)) << "\n";

		if (engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
			input_tensors.push_back(Tensor(name, i, dims, size));
		} else {
			output_tensors.push_back(Tensor(name, i, dims, size));
		}
	}

	std::cout << bindings_info.str();
}

void InferEngine::SetTensorAddr(std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
								std::vector<void*>& bindings, std::shared_ptr<nvinfer1::IExecutionContext>& context) {
	for (Tensor tensor : input_tensors) {
		context->setTensorAddress(tensor.name_.c_str(), bindings[tensor.index_]);
	}

	for (Tensor tensor : output_tensors) {
		context->setTensorAddress(tensor.name_.c_str(), bindings[tensor.index_]);
	}
}

void InferEngine::CudaMallocTensor(std::vector<void*>& bindings, int idx, size_t size) {
	cudaMalloc(&bindings[idx], size);
}

bool InferEngine::DeserializeEngine(std::shared_ptr<nvinfer1::ICudaEngine>& m_engine, std::string& model_path,
									mLogger& logger) {
	std::string engine_path = model_path + ".engine";
	std::ifstream engineFile(engine_path, std::ios::binary);
	if (engineFile.fail()) {
		std::cerr << "Engine read failed!\n";
		return false;
	} else {
		std::cout << "Engine read successful!\n";
	}

	engineFile.seekg(0, std::ifstream::end);
	auto fsize = engineFile.tellg();
	engineFile.seekg(0, std::ifstream::beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime_->deserializeCudaEngine(engineData.data(), fsize),
													  InferDeleter());
	engineFile.close();
	if (m_engine.get() != nullptr) {
		return true;
	} else {
		return false;
	}
}

size_t InferEngine::GetMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size) {
	return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}