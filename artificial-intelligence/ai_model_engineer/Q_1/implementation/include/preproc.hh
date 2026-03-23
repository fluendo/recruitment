#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

//  Tile dimensions  (Should be tuned to target GPU's shared-memory size
//  Each block handles a TILE_H x TILE_W patch of the image.
//  Must satisfy: TILE_H * TILE_W * C * sizeof(uint8) <= shared mem

#define TILE_W 32
#define TILE_H 32
#define CHANS 3

/**
 * @brief Host launcher 
 * 
 * @param d_src device source pointer HWC uint8 BGR
 * @param d_dst device destination pointer CHW float RGB
 * @param H height
 * @param W width
 * @param stream cuda stream obj
 */
void launch_preprocess(const uint8_t* d_src,  
					   float* d_dst,		  
					   int H, int W, cudaStream_t stream);