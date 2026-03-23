compute_capability=$(nvidia-smi --query-gpu=compute_cap --format=csv | awk -F'.' 'NR>1 {print $1$2}' | head -n 1)

cmake \
	-B build \
    -DCMAKE_CUDA_ARCHITECTURES=${compute_capability} \
	-DCMAKE_BUILD_TYPE=Release \
	-G Ninja

build_directory=$(realpath build/)
cmake --build ${build_directory} --config Release --target all -j $[ n=$(nproc), n<2?n:n-1 ]
