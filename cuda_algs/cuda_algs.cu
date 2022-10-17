#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h> 
#include <math.h>
#ifdef _WINDOWS
#include <conio.h>
#else
#define _cprintf printf
#endif

#include "cuda_algs.cuh"

static const int BLOCK_SIZE = 8;
bool bIsHaveCuda()
{
	//return false;
	int gpu_cnt;
	cudaGetDeviceCount(&gpu_cnt);
	if (gpu_cnt > 0) return true;
	return false;
}

__device__ float3 mat3x3_vec_mul(float* mat, float3 vec)
{
	float3 ret;
	ret.x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
	ret.y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
	ret.z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
	return ret;
}
__device__ float3 mat3x4_vec_mul(float* mat, float3 vec)
{
	float3 ret;
	ret.x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z + mat[3] * 1.0;
	ret.y = mat[4] * vec.x + mat[5] * vec.y + mat[6] * vec.z + mat[7] * 1.0;
	ret.z = mat[8] * vec.x + mat[9] * vec.y + mat[10] * vec.z + mat[11] * 1.0;
	return ret;
}

__device__ float3 cross(float3 a, float3 b)
{
	float3 c = make_float3(0, 0, 0);

	c.x = a.y * b.z - a.z * b.y;

	c.y = a.z * b.x - a.x * b.z;

	c.z = a.x * b.y - a.y * b.x;

	return c;
}
__device__ float length(float3 a)
{
	return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__device__ float3 unit(float3 a)
{
	float len = length(a);
	float3 ret = make_float3(a.x / len, a.y / len, a.z / len);
	return ret;
}

__device__ float3 minus(float3 a, float3 b)
{
	float3 ret = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	return ret;
}

__device__ float inner(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 plus(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 mul(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float clamp(float v, float min_v, float max_v)
{
	if (v < min_v) v = min_v;
	if (v > max_v) v = max_v;
	return v;
}


#define IMAGE_VAL_BPL(x,y,c,w,n,bpl, mem) (mem[x*n + y*bpl + c])
static float4 __device__ imageSamplingBPL(float xx, float yy, int c, int w, int h, int bpl, unsigned char* in)
{
	int x1 = (int)floor(xx);
	int x2 = (int)ceil(xx);
	int y1 = (int)floor(yy);
	int y2 = (int)ceil(yy);

	if (x1 < 0 || x1 >= w /*|| x2 < 0 || x2 >= w*/) return make_float4(0.0, 0.0, 0.0, 0.0);
	if (y1 < 0 || y1 >= h /*|| y2 < 0 || y2 >= h*/) return make_float4(0.0, 0.0, 0.0, 0.0);
	if (x2 >= w) {
		x2 = w - 1; xx = floor(xx);
	}
	if (y2 >= h) {
		y2 = h - 1; yy = floor(yy);
	}

	float4 ret = make_float4(0.0, 0.0, 0.0, 0.0);
	float v1, v2, v3, v4, tvx, tvy;

	tvx = xx - floor(xx);
	tvy = yy - floor(yy);

	for (int i = 0; i < c; i++) {
		v1 = float(IMAGE_VAL_BPL(x1, y1, i, w, c, bpl, in));
		v2 = float(IMAGE_VAL_BPL(x2, y1, i, w, c, bpl, in));
		v3 = float(IMAGE_VAL_BPL(x1, y2, i, w, c, bpl, in));
		v4 = float(IMAGE_VAL_BPL(x2, y2, i, w, c, bpl, in));
		v1 = (1.0 - tvx) * v1 + tvx * v2;
		v3 = (1.0 - tvx) * v3 + tvx * v4;
		v1 = (1.0 - tvy) * v1 + tvy * v3;
		if (i == 0) {
			ret.x = v1;
		}
		else if (i == 1) {
			ret.y = v1;
		}
		else if (i == 2) {
			ret.z = v1;
		}
		else {
			ret.w = v1;
		}
	}

	return ret;
}

__global__ void undistortKernal(unsigned char* input, unsigned char* output, int bytes_per_line, int width, int height,
	float2 focalLength, float2 principalPoint, float k1, float k2)
{
	const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y);

	int channel = bytes_per_line / width;
	if (uv_out.x >= width || uv_out.y >= height)
		return;

	const float u = uv_out.x;
	const float v = uv_out.y;

	const float _fx = 1.0f / focalLength.x;
	const float _fy = 1.0f / focalLength.y;

	const float y = (v - principalPoint.y) * _fy;
	const float y2 = y * y;

	const float x = (u - principalPoint.x) * _fx;
	const float x2 = x * x;
	const float r2 = x2 + y2;
	const float d = 1.0 + k1 * r2 + k2 * r2 * r2;
	const float _u = focalLength.x * x * d + principalPoint.x;
	const float _v = focalLength.y * y * d + principalPoint.y;

	const float2 uv_in = make_float2(_u, _v);

	if (uv_in.x >= float(width) || uv_in.y >= float(height) || uv_in.x < 0.0 || uv_in.y < 0.0)
		return;

	float4 outv = imageSamplingBPL(uv_in.x, uv_in.y, channel, width, height, bytes_per_line, input);
	for (int ch = 0; ch < channel; ch++) {
		output[uv_out.y * bytes_per_line + uv_out.x * channel + ch] =
			(unsigned char)(ch == 0 ? clamp(outv.x, 0.0, 255.0) : (ch == 1 ? clamp(outv.y, 0.0, 255.0) : (ch == 2 ? clamp(outv.z, 0.0, 255.0) : (ch == 3 ? clamp(outv.x, 0.0, 255.0) : 0))));
	}
}
void undistortFunc(unsigned char* input, int bytes_per_line, int width, int height,
	float fx, float fy, float cx, float cy, float k1, float k2)
{
	cudaError_t cudaStatus;

	unsigned char* d_input;
	cudaMalloc(&d_input, bytes_per_line * height);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		_cprintf("[CUDA] error malloc %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaMemcpy(d_input, input, bytes_per_line * height, cudaMemcpyKind::cudaMemcpyHostToDevice);
	unsigned char* d_output;
	cudaMalloc(&d_output, bytes_per_line * height);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		_cprintf("[CUDA] error malloc %s\n", cudaGetErrorString(cudaStatus));
	}
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid(ceil(float(width) / float(dim_block.x)), ceil(float(height) / float(dim_block.y)), 1);

	undistortKernal << < dim_grid, dim_block >> >
		(d_input, d_output, bytes_per_line, width, height, make_float2(fx, fy), make_float2(cx, cy), k1, k2);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		_cprintf("[CUDA] error undistort %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaMemcpy(input, d_output, bytes_per_line * height, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}

