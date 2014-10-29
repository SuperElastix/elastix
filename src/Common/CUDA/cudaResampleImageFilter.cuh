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
#ifndef __cudaCUDAResamplerImageFilter_h
#define __cudaCUDAResamplerImageFilter_h

#include <cuda_runtime.h>
#include "cudaMacro.h"

namespace cuda
{

template <class TInputImageType, class TOutputImageType>
TOutputImageType* cudaCastToType( const cudaExtent & volumeExtent,
  const TInputImageType* src, TOutputImageType* dst,
  cudaMemcpyKind direction, bool useCPU );

// Specialization
template <>
float* cudaCastToType( const cudaExtent & volumeExtent,
  const float* src, float* dst,
  cudaMemcpyKind direction, bool useCPU );

/**
 * Helper class
 */

class cudaTextures
{
public:
  /* Linear mode filtering - which we need - is only supported for floating-point types. */
  static const enum cudaTextureFilterMode cudaFilterMode = cudaFilterModeLinear;

#if defined(__CUDACC__)
  typedef texture<float, 3, cudaReadModeElementType> texture_3D_t; /* 3D texture */
#endif /* __CUDACC__ */
};


/** \class CUDAResampleImageFilter
 * \brief Resample an image on the GPU via a coordinate transform
 *
 * This class really implements the CUDA resampling. It should be compiled
 * by the CUDA compiler.
 *
 * \ingroup GeometricTransforms
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
class CUDAResampleImageFilter:
  public cudaTextures
{
public:

  CUDAResampleImageFilter();
  ~CUDAResampleImageFilter();

  void cudaInit( void );
  void cudaUnInit( void );

  void cudaCopyImageSymbols( const float3 & InputImageSpacing, const float3 & InputImageOrigin,
    const float3 & OutputImageSpacing, const float3 & OutputImageOrigin, const float DefaultPixelValue );
  void cudaCopyGridSymbols( const float3 & GridSpacing, const float3 & GridOrigin, const uint3 & GridSize );

  void cudaMallocTransformationData( const uint3 & gridSize, const TInterpolatorPrecisionType* params );
  void cudaMallocImageData( const uint3 & inputsize, const uint3 & outputsize, const TImageType* data );

  void cudaCastToHost( const size_t sizevalue, const TInternalImageType* src,
    TInternalImageType* tmp_src, TImageType* dst );
  void cudaCastToHost( const uint3 & sizevalue, const TInternalImageType* src, TImageType* dst );
  void cudaCastToDevice( const uint3 & sizevalue, const TImageType* src, TInternalImageType* dst );

  void GenerateData( TImageType* dst );

  cudaGetConstMacro( OutputImage, TInternalImageType* );
  cudaGetConstMacro( OutputImageSize, uint3 );

  cudaGetConstMacro( Device, int );
  cudaSetMacro( Device, int );

  cudaSetMacro( CastOnGPU, bool );
  cudaGetConstMacro( CastOnGPU, bool );

  cudaSetMacro( UseFastCUDAKernel, bool );
  cudaGetConstMacro( UseFastCUDAKernel, bool );

  static int checkExecutionParameters( void );

private:

  /** Private member variables. */
  cudaArray*      m_CoeffsX;
  cudaArray*      m_CoeffsY;
  cudaArray*      m_CoeffsZ;
  cudaArray*      m_InputImage;
  TInternalImageType*   m_OutputImage;
  uint3           m_InputImageSize;
  uint3           m_OutputImageSize;
  size_t          m_NumberOfInputVoxels;
  size_t          m_NumberOfOutputVoxels;
  cudaChannelFormatDesc m_ChannelDescCoeff;
  int             m_Device;
  bool            m_CastOnGPU;
  bool            m_UseFastCUDAKernel;

  unsigned int    m_MaxNumberOfVoxelsPerIteration;

#if defined(__CUDACC__)
  template <typename tex_t> cudaError_t cudaBindTextureToArray(
    cudaArray* dst, const TInternalImageType* src,
    const cudaExtent & extent, tex_t & tex, cudaChannelFormatDesc & desc,
    bool normalized = false, bool onDevice = false );
#endif /* __CUDACC__ */

}; // end class CUDAResampleImageFilter

}; // end namespace cuda

#endif // end #ifndef __cudaCUDAResamplerImageFilter_h
