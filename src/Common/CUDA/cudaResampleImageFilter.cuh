/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __cudaCUDAResamplerImageFilter_h
#define __cudaCUDAResamplerImageFilter_h

#include <cuda_runtime.h>
#include "cudaMacro.h"

namespace cuda
{

template <class TInputImageType, class TOutputImageType>
TOutputImageType* cudaCastToType( cudaExtent& volumeExtent,
  const TInputImageType* src, TOutputImageType* dst,
  cudaMemcpyKind direction, bool UseCPU );

// Specialization
template <>
float* cudaCastToType( cudaExtent& volumeExtent,
  const float* src, float* dst,
  cudaMemcpyKind direction, bool UseCPU );

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

  void cudaCopyImageSymbols( float3& InputImageSpacing, float3& InputImageOrigin,
    float3& OutputImageSpacing, float3& OutputImageOrigin, float DefaultPixelValue );
  void cudaCopyGridSymbols( float3& GridSpacing, float3& GridOrigin, int3& GridSize );

  void cudaMallocTransformationData( int3 gridSize, const TInterpolatorPrecisionType* params );
  void cudaMallocImageData( int3 inputsize, int3 outputsize, const TImageType* data );

  void cudaCastToHost( size_t size, const TInternalImageType* src, TInternalImageType* tmp_src, TImageType* dst );
  void cudaCastToHost( int3 size, const TInternalImageType* src, TImageType* dst );
  void cudaCastToDevice( int3 size, const TImageType* src, TInternalImageType* dst );

  void GenerateData( TImageType* dst );

  cudaGetConstMacro( OutputImage, TInternalImageType* );
  cudaGetConstMacro( OutputImageSize, int3 );
  cudaGetConstMacro( Device, int );
  cudaSetMacro( Device, int );

  cudaSetMacro( CastOnGPU, bool );
  cudaGetConstMacro( CastOnGPU, bool );

  static int checkExecutionParameters( void );

private:

  /** Private member variables. */
  cudaArray*      m_coeffsX;
  cudaArray*      m_coeffsY;
  cudaArray*      m_coeffsZ;
  cudaArray*      m_InputImage;
  TInternalImageType*   m_OutputImage;
  int3            m_InputImageSize;
  int3            m_OutputImageSize;
  size_t          m_nrOfInputVoxels;
  size_t          m_nrOfOutputVoxels;
  cudaChannelFormatDesc m_channelDescCoeff;
  int             m_Device;
  bool            m_CastOnGPU;

  unsigned int    m_MaxnrOfVoxelsPerIteration;

#if defined(__CUDACC__)
  template <typename tex_t> cudaError_t cudaBindTextureToArray(
    cudaArray* dst, const TInternalImageType* src,
    cudaExtent& extent, tex_t& tex, cudaChannelFormatDesc& desc,
    bool normalized = false, bool onDevice = false );
#endif /* __CUDACC__ */

}; // end class CUDAResampleImageFilter

}; // end namespace cuda

#endif // end #ifndef __cudaCUDAResamplerImageFilter_h
