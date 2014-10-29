/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
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

/* Convert an index that is an offset to a 3D matrix into its xyz coordinates */
__device__ __host__ uint3 index2coord( const unsigned int index, const uint3 dimension )
{
  /** WARNING: Direction is not yet taken into account! */

  int tmp = dimension.x * dimension.y;
  uint3 res;
  res.z = index / tmp;
  tmp = index - (res.z * tmp);

  res.y = tmp / dimension.x;
  res.x = tmp - (res.y * dimension.x);
  return res;
}

/* Apply a 3D B-spline transformation on a coordinate. */
__device__ float3 deform_at_coord( float3 coord )
{
  float3 res;
  /** Coordinate shift, since appearantly the space Ruijters lives in is
   * a little shifted from our space.
   */
  coord += 0.5f;

  /** A B-spline transformation is seperable among its dimensions! */
  res.x = cubicTex3D( m_tex_coeffsX, coord );
  res.y = cubicTex3D( m_tex_coeffsY, coord );
  res.z = cubicTex3D( m_tex_coeffsZ, coord );

  return res;
}

__device__ float3 deform_at_coord_simple( float3 coord )
{
  float3 res;
  /** Coordinate shift, since appearantly the space Ruijters lives in is
   * a little shifted from our space.
   */
  coord += 0.5f;

  /** A B-spline transformation is seperable among its dimensions! */
  res.x = cubicTex3DSimple( m_tex_coeffsX, coord );
  res.y = cubicTex3DSimple( m_tex_coeffsY, coord );
  res.z = cubicTex3DSimple( m_tex_coeffsZ, coord );

  return res;
}

/* Apply deformation to all voxels based on transform parameters and retrieve result. */
template <typename TImageType>
__global__ void resample_image( TImageType* dst,
  uint3 inputImageSize, uint3 outputImageSize, size_t offset, bool useFastKernel = false )
{
  size_t id = threadIdx.x + ( blockIdx.x * blockDim.x );

  /* Convert single index to coordinates. */
  uint3 coord = index2coord( id + offset, outputImageSize );
  float3 out_coord = make_float3( coord.x, coord.y, coord.z );

  /* Translate normal coordinates into world coordinates.
   * WARNING: Direction is not yet taken into account!
   */
  float3 out_coord_world = out_coord * CUOutputImageSpacing + CUOutputImageOrigin;

  /* Translate world coordinates in terms of B-spline grid. */
  float3 out_coord_world_bspline = ( out_coord_world - CUGridOrigin ) / CUGridSpacing;

  /* Check if the sample is within the B-spline grid. */
  bool isValidSample = ( out_coord_world_bspline >= 0.0f
    && out_coord_world_bspline < make_float3( CUGridSize - 2 ) );
  float res = CUDefaultPixelValue;

  if ( isValidSample )
  {
    /* B-Spline deform of a coordinate uses world coordinates. */
    float3 deform = deform_at_coord( out_coord_world_bspline );
    float3 inp_coord_world = out_coord_world + deform;

    /* Translate world coordinates to normal coordinates.
     * WARNING: Direction is not yet taken into account!
     */
    float3 inp_coord = ( (inp_coord_world - CUInputImageOrigin) / CUInputImageSpacing ) + 0.5f;

    /** Check if sample is inside input image. */
    isValidSample = ( inp_coord > 0.0f )
      && inp_coord < make_float3( inputImageSize.x, inputImageSize.y, inputImageSize.z );

    /* Interpolate the moving/input image using 3-rd order B-spline. */
    if ( isValidSample )
    {
      if ( useFastKernel )
      {
        res = cubicTex3D( m_tex_inputImage, inp_coord );
      }
      else
      {
        res = cubicTex3DSimple( m_tex_inputImage, inp_coord );
      }
    }
  }

  dst[ id ] = static_cast<TImageType>( res );
}

/* Cast from one type to another type on the GPU. */
template <class TInputImageType, class TOutputImageType>
__global__ void cast_to_type( TOutputImageType* dst,
  const TInputImageType* src, size_t nrOfVoxels )
{
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
  if ( id >= nrOfVoxels ) return;

  dst[ id ] = (TOutputImageType)src[ id ];
  //dst[ id ] = static_cast<TOutputImageType>( src[ id ] );
}
