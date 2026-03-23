// Fused BGR->RGB, HWC->CHW transpose + normalize for YOLOv8 input
// Input:  uint8 HWC [H][W][C]  (e.g. 640x640x3, BGR from OpenCV)
// Output: float CHW [C][H][W]  (e.g. 3x640x640, RGB, /255.0)
#include "preproc.hh"

/**
 * @brief Converts image data from HWC BGR uint8 to CHW RGB float layout,
 *        with [0, 1] normalization.
 *
 * @param[in]  src  Input image in HWC BGR uint8 format, shape [H][W][3]
 * @param[out] dst  Output image in CHW RGB float format, shape [3][H][W], normalized to [0, 1]
 * @param[in]  H    Image height in pixels
 * @param[in]  W    Image width in pixels
 *
 * @par Grid
 *   `(ceil(W/TILE_W), ceil(H/TILE_H), 1)`
 *
 * @par Block
 *   `(TILE_W, TILE_H, 1)`
 *
 * @par SIMT Design
 * - Every thread in the warp follows the same control path
 * - Phase 1 — coalesced HWC global->shared load: threads in the same warp
 *   have consecutive `threadIdx.x`, so source addresses are stride-3 apart;
 *   out-of-bounds threads write 0 to shared memory
 * - Shared memory store: bank-conflict-free via +1 column padding
 * - Phase 2 — coalesced CHW shared→global write: stride-1 in W for a
 *   fixed channel and row; BGR→RGB channel flip and x/255 normalization
 *   applied on write-out
 */
__global__ void hwc_to_chw_normalize(const uint8_t* __restrict__ src,  // HWC uint8  BGR
									 float* __restrict__ dst,		   // CHW float  RGB
									 int H, int W) {
	// Pad by 1 in W to avoid shared-memory bank conflicts on load
	// 3 channels x (TILE_H rows x (TILE_W+1) cols)
	__shared__ uint8_t tile[CHANS][TILE_H][TILE_W + 1];

	// pixel coordinates
	int col = blockIdx.x * TILE_W + threadIdx.x;  // W dimension
	int row = blockIdx.y * TILE_H + threadIdx.y;  // H dimension

	// Phase 1: coalesced global->shared load
	// Threads in the same warp have consecutive threadIdx.x,
	// so their HWC source addresses are stride-3 apart (one pixel).
	// We unroll over C so each thread loads all 3 bytes of its pixel.
	// Within a warp (same row), loads are 3 bytes apart -> coalesced
	// when the memory transaction covers the 32-thread stride.
	if (row < H && col < W) {
		int src_idx = (row * W + col) * CHANS;	// HWC offset
// Load BGR; we will flip to RGB on write-out
#pragma unroll
		for (int c = 0; c < CHANS; ++c) tile[c][threadIdx.y][threadIdx.x] = src[src_idx + c];
	} else {
// Out-of-bounds threads write 0 (safe for padding regions)
#pragma unroll
		for (int c = 0; c < CHANS; ++c) tile[c][threadIdx.y][threadIdx.x] = 0;
	}

	// Barrier: all threads in block have loaded their pixel
	__syncthreads();

	// Phase 2: shared->global write (CHW, normalized, BGR->RGB)
	// dst layout: [C][H][W] -> flat index = c*H*W + row*W + col
	// Threads in the same warp still have consecutive col values
	// -> writes are coalesced (stride-1 in W for a fixed c,row).
	if (row < H && col < W) {
		int chw_base = row * W + col;  // spatial offset, same for all C
		float inv255 = 1.0f / 255.0f;

		// BGR (c=0→B, c=1->G, c=2->R) -> RGB (out_c=0->R, 1->G, 2->B)
		int bgr_to_rgb[3] = {2, 1, 0};	// src channel mapping

#pragma unroll
		for (int out_c = 0; out_c < CHANS; ++out_c) {
			int in_c = bgr_to_rgb[out_c];
			float val = tile[in_c][threadIdx.y][threadIdx.x] * inv255;
			dst[out_c * H * W + chw_base] = val;
		}
	}
}


void launch_preprocess(const uint8_t* d_src,  // device ptr, HWC uint8 BGR
					   float* d_dst,		  // device ptr, CHW float RGB
					   int H, int W, cudaStream_t stream) {
	dim3 block(TILE_W, TILE_H, 1);
	dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, 1);
	hwc_to_chw_normalize<<<grid, block, 0, stream>>>(d_src, d_dst, H, W);
}
