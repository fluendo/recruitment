#include "FaceDetectEngine.hh"
#include "ReIdEngine.hh"
#include <opencv2/opencv.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void saveGallery(const std::unordered_map<std::string, std::vector<float>>& gallery, const std::string& path) {
	json j;
	for (const auto& [id, emb] : gallery) j[id] = emb;

	std::ofstream f(path);
	f << j.dump(2);
}

std::unordered_map<std::string, std::vector<float>> loadGallery(const std::string& path) {
	std::unordered_map<std::string, std::vector<float>> gallery;

	std::ifstream f(path);
	if (!f.is_open()) {
		std::ofstream create(path);
		create << "{}";
		return gallery;
	}

	f.seekg(0, std::ios::end);
	if (f.tellg() == 0) return gallery;
	f.seekg(0, std::ios::beg);

	json j = json::parse(f);
	for (const auto& [id, emb] : j.items()) gallery[id] = emb.get<std::vector<float>>();

	return gallery;
}

void PrintBoxes(const std::vector<FaceSegm>& bound_boxes) {
	fmt::print("Detected {} face(s)\n", bound_boxes.size());
	for (size_t i = 0; i < bound_boxes.size(); ++i) {
		const auto& fs = bound_boxes[i];
		if (fs.conf > 0) {
			fmt::print("[{}] cx={} cy={} w={} h={} conf={:.3f}\n", i, fs.cx, fs.cy, fs.w, fs.h, fs.conf);
		}
	}
}

cv::Mat letterbox(const cv::Mat& src, int target = 640) {
	int max_dim = std::max(src.cols, src.rows);
	cv::Mat canvas = cv::Mat::zeros(max_dim, max_dim, src.type());
	src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
	cv::Mat dst;
	cv::resize(canvas, dst, cv::Size(target, target));
	return dst;
}

std::vector<float> normalize(const std::vector<float>& v) {
	std::vector<float> out(v.size());
	float norm = 0.f;
	for (int k = 0; k < (int)v.size(); ++k) norm += v[k] * v[k];
	norm = std::sqrt(norm);
	if (norm > 1e-6f)
		for (int k = 0; k < (int)v.size(); ++k) out[k] = v[k] / norm;
	return out;
}

int main(int argc, char** argv) {
	std::cout << "Starting\n";
	FaceDetectEngine segment(
		std::string(
			"/wp/fluendo/artificial-intelligence/ai_model_engineer/Q_1/implementation/onnx/yolov8n-face-lindevs"),
		0);
	Dimensions dims_in(640, 640, 3);
	segment.InitInfer(dims_in);

	ReIdEngine reid(
		std::string("/wp/fluendo/artificial-intelligence/ai_model_engineer/Q_1/implementation/onnx/arcface"), 0);
	Dimensions dims_in_reid(112, 112, 3);
	reid.InitInfer(dims_in_reid);

	std::cout << "Start infer\n";
	std::string imagePath =
		"/wp/fluendo/artificial-intelligence/ai_model_engineer/Q_1/implementation/resources/young-person-intership.png";
	float thresh = 0.1;

	// Keep original untouched — clone fresh every iteration for drawing
	const cv::Mat image_orig = cv::imread(imagePath, cv::IMREAD_COLOR);
	cv::Mat res = letterbox(image_orig);

	std::vector<float> host_ptr_res(5 * 8400, 0);
	std::vector<FaceSegm> bound_boxes;

	const std::string galleryPath =
		"/wp/fluendo/artificial-intelligence/ai_model_engineer/Q_1/implementation/resources/gallery.json";
	auto gallery = loadGallery(galleryPath);

	// First launch: gallery is empty → enroll mode
	// Second launch: gallery has person_0 → verify mode
	const bool enroll_mode = gallery.empty();
	fmt::print("{}\n", enroll_mode ? "Mode: ENROLL" : "Mode: VERIFY");

	std::vector<std::vector<float>> embeddings_vec;
	long long total_ms = 0;
	int iter_count = 0;
	cv::Mat output_image;

	for (int i = 0; i < 50; i++) {
		auto start = std::chrono::steady_clock::now();

		// Fresh drawable copy every iteration - image_orig is never touched
		cv::Mat image = image_orig.clone();

		segment.Infer(res.ptr<uint8_t>(0), host_ptr_res.data());
		segment.DecodeRes(host_ptr_res.data(), bound_boxes, 1920, 1080, thresh);
		PrintBoxes(bound_boxes);

		std::vector<cv::Mat> faces;
		embeddings_vec.clear();

		// Crop from original 
		for (auto bb : bound_boxes) {
			cv::Rect roi(bb.cx - bb.w / 2, bb.cy - bb.h / 2, bb.w, bb.h);
			faces.push_back(image_orig(roi).clone());
			embeddings_vec.push_back(std::vector<float>(512, 0.0));
		}

		for (int fi = 0; fi < (int)faces.size(); ++fi) {
			cv::Mat roi_resized, roi_rgb, roi_float;
			cv::resize(faces[fi], roi_resized, cv::Size(112, 112));
			cv::cvtColor(roi_resized, roi_rgb, cv::COLOR_BGR2RGB);
			roi_rgb.convertTo(roi_float, CV_32FC3, 1.0 / 128.0, -127.5 / 128.0);
			reid.Infer((float*)roi_float.data, embeddings_vec[fi].data());
		}

		if (i < 15) continue;

		if (!enroll_mode) {
			for (int j = 0; j < (int)embeddings_vec.size(); ++j) {
				std::string best_id = "unknown";
				float best_sim = -1.f;
				const float threshold = 0.95f;

				const auto query = normalize(embeddings_vec[j]);
				int dim = (int)query.size();

				for (const auto& [id, emb] : gallery) {
					const auto gallery_emb = normalize(emb);

					float dot = 0.f;
					for (int k = 0; k < dim; ++k) dot += query[k] * gallery_emb[k];

					if (dot > best_sim) {
						best_sim = dot;
						best_id = (dot >= threshold) ? id : "unknown";
					}
				}

				fmt::print("Face {} → {} (sim={:.3f})\n", j, best_id, best_sim);

				const auto& bb = bound_boxes[j];
				cv::Rect roi(bb.cx - bb.w / 2, bb.cy - bb.h / 2, bb.w, bb.h);

				if (best_id == "unknown") {
					cv::Mat face_roi = image(roi);
					cv::GaussianBlur(face_roi, face_roi, cv::Size(51, 51), 0);
				} else {
					cv::Mat overlay = image.clone();
					cv::rectangle(overlay, roi, cv::Scalar(0, 255, 0), -1);
					cv::addWeighted(overlay, 0.3, image, 0.7, 0, image);
					cv::putText(image, best_id, cv::Point(bb.cx - bb.w / 2, bb.cy - bb.h / 2 - 5),
								cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
				}
			}
		}

		output_image = image;  // keep last rendered frame

		auto end = std::chrono::steady_clock::now();
		auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Infer time " << total.count() << " ms\n";
		total_ms += total.count();
		++iter_count;
	}

	fmt::print("Average infer time: {:.1f} ms\n", iter_count > 0 ? (double)total_ms / iter_count : 0.0);

	cv::imwrite("/wp/fluendo/artificial-intelligence/ai_model_engineer/Q_1/implementation/resources/result.png",
				output_image.empty() ? image_orig : output_image);

	// On first launch: save person_0 embedding and exit
	if (enroll_mode) {
		if (!embeddings_vec.empty()) {
			gallery["person_0"] = embeddings_vec[0];
			fmt::print("Enrolled person_0 emb[0]={:.4f}\n", embeddings_vec[0][0]);
			saveGallery(gallery, galleryPath);
		} else {
			fmt::print("No faces detected, nothing enrolled.\n");
		}
	}
}