/* $Id$ */
#include "CI/cubicTex3D.cu"

cuda::cudaTextures::texture_3D_t m_tex_coeffsX;
cuda::cudaTextures::texture_3D_t m_tex_coeffsY;
cuda::cudaTextures::texture_3D_t m_tex_coeffsZ;
cuda::cudaTextures::texture_3D_t m_tex_inputImage;

__device__ bool operator<(float3 a, float3 b)
{
	return a.x < b.x && a.y < b.y && a.z < b.z;
}

__device__ bool operator>(float3 a, float b)
{
	return a.x > b && a.y > b && a.z > b;
}

__device__ bool operator<(float3 a, float b)
{
	return a.x < b && a.y < b && a.z < b;
}

__device__ bool operator>=(float3 a, float b)
{
	return a.x >= b && a.y >= b && a.z >= b;
}

__device__ bool operator>=(float3 a, float3 b)
{
	return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

__device__ int3 operator-(int3 a, int b)
{
	return make_int3(a.x - b, a.y - b, a.z - b);
}

__device__ void operator+=(float3& a, float b)
{
	a.x += b; a.y += b; a.z += b;
}

/* convert an index that is an offset to a 3D matrix into its xyz coordinates */
__device__ __host__ int3 index2coord(int index, const int3 DIM)
{
	int tmp = DIM.x * DIM.y;
	int3 res;
	res.z = index / tmp;
	tmp = index - (res.z * tmp);

	res.y = tmp / DIM.x;
	res.x = tmp - (res.y * DIM.x);
	return res;
}

/* apply a 3D B-spline registration on a coordinate */
__device__ float3 deform_at_coord(float3 coord)
{
	float3 res;
	coord += 0.5f;
	res.x = cubicTex3D(m_tex_coeffsX, coord);
	res.y = cubicTex3D(m_tex_coeffsY, coord);
	res.z = cubicTex3D(m_tex_coeffsZ, coord);
	return res;
}

__device__ float3 deform_at_coord_simple(float3 coord)
{
	float3 res;
	coord += 0.5f;
	res.x = cubicTex3DSimple(m_tex_coeffsX, coord);
	res.y = cubicTex3DSimple(m_tex_coeffsY, coord);
	res.z = cubicTex3DSimple(m_tex_coeffsZ, coord);
	return res;
}

/* apply deformation to all voxels based on transform parameters and retrieve result */
template <typename TImageType>
__global__ void resample_image(TImageType* dst, int3 inputImageSize, int3 outputImageSize, size_t offset)
{
	size_t id = threadIdx.x + (blockIdx.x * blockDim.x);

	/* convert single index to coordinates */
	int3 coord = index2coord(id + offset, outputImageSize);

	float3 out_coord               = make_float3(coord.x, coord.y, coord.z);
	/* translate normal coordinates into world coordinates */
	float3 out_coord_world         = out_coord * CUOutputImageSpacing + CUOutputImageOrigin;
	/* translate world coordinates in terms of B-spline grid */
	float3 out_coord_world_bspline = (out_coord_world - CUGridOrigin) / CUGridSpacing;

	/* check if within B-spline grid */
	bool isValidSample = (out_coord_world_bspline >= 0.0f && out_coord_world_bspline < make_float3(CUGridSize - 2));
	float res = CUDefaultPixelValue;
	
	if (isValidSample) {
		/* B-Spline deform of a coordinate uses world coordinate */
		float3 deform = deform_at_coord_simple(out_coord_world_bspline);
		float3 inp_coord_world = out_coord_world + deform;
		/* translate world coordinates to normal coordinates */
		float3 inp_coord = ((inp_coord_world - CUInputImageOrigin) / CUInputImageSpacing);

		isValidSample = (inp_coord > 0.0f) && inp_coord < make_float3(inputImageSize - 1);

		/* B-spline transform of a coordinate uses normal coordinates */
		if (isValidSample) res = cubicTex3DSimple(m_tex_inputImage, inp_coord + 0.5f);
	}
	
	dst[id] = static_cast<TImageType>(res);
}

/* cast from one type to another type on the GPU */
template <class TInputImageType, class TOutputImageType>
__global__ void cast_to_type(TOutputImageType* dst, const TInputImageType* src, size_t nrOfVoxels)
{
	int id = threadIdx.x + (blockIdx.x * blockDim.x);
	if (id >= nrOfVoxels) return;

	dst[id] = (TOutputImageType)src[id];
}
