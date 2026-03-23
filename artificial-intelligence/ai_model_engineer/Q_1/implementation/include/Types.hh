#pragma once

struct Dimensions {
	int width;	  /**< Width of the object*/
	int height;	  /**< Height of the object*/
	int channels; /**<  Number of channels */

	/**
	 * @brief Default constructor
	 *
	 */
	Dimensions() : width(0), height(0), channels(0) {}

	/**
	 * @brief Construct a new Dimensions object
	 *
	 * @param w width
	 * @param h height
	 * @param c channels
	 */
	Dimensions(int w, int h, int c) : width(w), height(h), channels(c) {}

	/**
	 * @brief Returns total size
	 *
	 * @return int
	 */
	int totalSize() const { return width * height * channels; }
};

/**
 * @brief Bounding box structure
 * 
 */
struct FaceSegm {
	int cx, cy, w, h;
	float conf;
};