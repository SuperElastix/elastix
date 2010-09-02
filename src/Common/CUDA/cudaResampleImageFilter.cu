/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "cudaResampleImageFilter.cuh"
#include "CI/cubicPrefilter3D.cu"
#include "cudaInlineFunctions.h"


__constant__ float3 CUInputImageSpacing;
__constant__ float3 CUInputImageOrigin;
__constant__ float3 CUOutputImageSpacing;
__constant__ float3 CUOutputImageOrigin;
__constant__ float3 CUGridSpacing;
__constant__ float3 CUGridOrigin;
__constant__ int3   CUGridSize;
__constant__ float  CUDefaultPixelValue;


#include "cudaDeformationsKernel.cu"

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::CUDAResampleImageFilter()
  : m_coeffsX(NULL)
  , m_coeffsY(NULL)
  , m_coeffsZ(NULL)
  , m_InputImage (NULL)
  , m_InputImageSize(make_int3(0,0,0))
  , m_Device(0)
  , m_CastOnGPU(false)
  , m_MaxnrOfVoxelsPerIteration(1 << 20)
{
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::~CUDAResampleImageFilter()
{
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaInit( void )
{
  checkExecutionParameters();
  cuda::cudaSetDevice( m_Device );
  m_channelDescCoeff = cudaCreateChannelDesc<TInternalImageType>();
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaUnInit()
{
  cuda::cudaUnbindTexture(m_tex_coeffsX);
  cuda::cudaUnbindTexture(m_tex_coeffsY);
  cuda::cudaUnbindTexture(m_tex_coeffsZ);
  cuda::cudaUnbindTexture(m_tex_inputImage);
  cuda::cudaFreeArray(m_coeffsX);
  cuda::cudaFreeArray(m_coeffsY);
  cuda::cudaFreeArray(m_coeffsZ);
  cuda::cudaFreeArray(m_InputImage);
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
int
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::checkExecutionParameters()
{
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount( &deviceCount );
  return ( err == cudaSuccess ) ? ( deviceCount == 0 ) : 1;
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCopyImageSymbols(
  float3& InputImageSpacing, float3& InputImageOrigin,
  float3& OutputImageSpacing, float3& OutputImageOrigin,
  float DefaultPixelValue )
{
  /* Copy some constant parameters to the GPU's constant cache. */
  cuda::cudaMemcpyToSymbol( CUInputImageSpacing,  InputImageSpacing,  cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUInputImageOrigin,   InputImageOrigin,   cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUOutputImageSpacing, OutputImageSpacing, cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUOutputImageOrigin,  OutputImageOrigin,  cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUDefaultPixelValue,  DefaultPixelValue,  cudaMemcpyHostToDevice );
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCopyGridSymbols( float3& GridSpacing, float3& GridOrigin, int3& GridSize )
{
  /* Copy some constant parameters to the GPU's constant cache. */
  cuda::cudaMemcpyToSymbol( CUGridSpacing,  GridSpacing,  cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUGridOrigin,   GridOrigin,   cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUGridSize,     GridSize,     cudaMemcpyHostToDevice );
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaMallocTransformationData( int3 gridSize, const TInterpolatorPrecisionType* params )
{
  const int nrOfParametersPerDimension = gridSize.x * gridSize.y * gridSize.z;
  cudaExtent gridExtent = make_cudaExtent(gridSize.x, gridSize.y, gridSize.z);

  /* Allocate memory on the GPU for the interpolation texture. */
  cuda::cudaMalloc3DArray( &m_coeffsX, &m_channelDescCoeff, gridExtent );
  cuda::cudaMalloc3DArray( &m_coeffsY, &m_channelDescCoeff, gridExtent );
  cuda::cudaMalloc3DArray( &m_coeffsZ, &m_channelDescCoeff, gridExtent );

  /* Convert TInterpolatorPrecisionType to float, only thing textures support. */
#if 1
  //clock_t start = clock();
  TInternalImageType* params_tmp = new TInternalImageType[ nrOfParametersPerDimension * 3 ];
  for ( size_t i = 0; i != nrOfParametersPerDimension * 3; ++i )
  {
    params_tmp[ i ] = static_cast<TInternalImageType>( params[ i ] );
  }
  //std::cout << "parameter type conversion took " << clock() - start << "ms for " << nrOfParametersPerDimension * 3 << " elements" << std::endl;
  cudaBindTextureToArray( m_coeffsX, &params_tmp[ 0 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsX, m_channelDescCoeff );
  cudaBindTextureToArray( m_coeffsY, &params_tmp[ 1 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsY, m_channelDescCoeff );
  cudaBindTextureToArray( m_coeffsZ, &params_tmp[ 2 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsZ, m_channelDescCoeff );
  delete[] params_tmp;
#else
  /* There are some problems with Device2Device copy when src is not a pitched or 3D array. */
  TInternalImageType* params_gpu
    = cuda::cudaMalloc<TInternalImageType>( nrOfParametersPerDimension );

  /* Create the B-spline coefficients texture. */
  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 0 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_coeffsX, params_gpu, gridExtent, m_tex_coeffsX,
    m_channelDescCoeff, false, true );

  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 1 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_coeffsY, params_gpu, gridExtent, m_tex_coeffsY,
    m_channelDescCoeff, false, true );

  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 2 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_coeffsZ, params_gpu, gridExtent, m_tex_coeffsZ,
    m_channelDescCoeff, false, true );

  cuda::cudaFree( params_gpu );
#endif
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaMallocImageData( int3 inputsize, int3 outputsize, const TImageType* data )
{
  m_InputImageSize        = inputsize;
  m_OutputImageSize       = outputsize;
  m_nrOfInputVoxels       = m_InputImageSize.x  * m_InputImageSize.y  * m_InputImageSize.z;
  size_t nrOfOutputVoxels = m_OutputImageSize.x * m_OutputImageSize.y * m_OutputImageSize.z;
  m_MaxnrOfVoxelsPerIteration = min( (unsigned int)nrOfOutputVoxels, m_MaxnrOfVoxelsPerIteration );

  cudaExtent volumeExtent = make_cudaExtent(
    m_InputImageSize.x, m_InputImageSize.y, m_InputImageSize.z );

  /* Allocate in memory and PreFilter image. We need to cast to float if not
   *already, because linear filtering only works with floating point values.
   */
  TInternalImageType* inputImage = cuda::cudaMalloc<TInternalImageType>( m_nrOfInputVoxels );
  cudaCastToDevice( m_InputImageSize, data, inputImage );
  CubicBSplinePrefilter3D( inputImage,
    volumeExtent.width, volumeExtent.height, volumeExtent.depth );

  /* XXX - cudaMemcpy3D fails if a DeviceToDevice copy src is not allocated
   * with cudaMallocPitch or cudaMalloc3D, so we need this hack to get the data there.
   */
  TInternalImageType* tmpImage = new TInternalImageType[ m_nrOfInputVoxels ];
  cuda::cudaMemcpy( tmpImage, inputImage, m_nrOfInputVoxels, cudaMemcpyDeviceToHost );
  cuda::cudaFree( inputImage );

  /* Create the image interpolation texture. */
  cuda::cudaMalloc3DArray( &m_InputImage, &m_channelDescCoeff, volumeExtent );
  cudaBindTextureToArray( m_InputImage, tmpImage, volumeExtent, m_tex_inputImage, m_channelDescCoeff );
  delete[] tmpImage;

  /* Allocate destination array. */
  m_OutputImage = cuda::cudaMalloc<TInternalImageType>( m_MaxnrOfVoxelsPerIteration );
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::GenerateData( TImageType* dst )
{
  /* Split up applying the transformation due to memory constraints and make
   * sure we never overflow the output image dimensions.
   */
  const size_t nrOfOutputVoxels = m_OutputImageSize.x * m_OutputImageSize.y * m_OutputImageSize.z;
  dim3 dimBlock( 256 );
  dim3 dimGrid( m_MaxnrOfVoxelsPerIteration / dimBlock.x );
  size_t offset = 0;

  TInternalImageType* tmp_src = new TInternalImageType[ m_MaxnrOfVoxelsPerIteration ];
  if ( nrOfOutputVoxels > m_MaxnrOfVoxelsPerIteration )
  {
    /* Do a full run of m_MaxnrOfVoxelsPerIteration voxels. */
    for ( offset = 0; offset <= nrOfOutputVoxels - m_MaxnrOfVoxelsPerIteration;
      offset += m_MaxnrOfVoxelsPerIteration )
    {
      resample_image<<<dimGrid, dimBlock>>>( m_OutputImage,
        m_InputImageSize, m_OutputImageSize, offset );
      cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
      cudaCastToHost( m_MaxnrOfVoxelsPerIteration, m_OutputImage, tmp_src, &dst[offset] );
    }
  }

  /* Do the remainder ensuring again dimGrid * dimBlock is less than image size. */
  dimGrid = dim3((unsigned int)(nrOfOutputVoxels - offset)) / dimBlock;
  resample_image<<<dimGrid, dimBlock>>>( m_OutputImage,
    m_InputImageSize, m_OutputImageSize, offset );
  cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
  cudaCastToHost( dimGrid.x * dimBlock.x, m_OutputImage, tmp_src, &dst[offset] );

  /* Do the final amount of voxels < dimBlock. */
  offset += dimGrid.x * dimBlock.x;
  dimBlock = dim3((unsigned int)(nrOfOutputVoxels - offset));
  dimGrid  = dim3( 1 );

  if ( dimBlock.x > 0 )
  {
    resample_image<<<dimGrid, dimBlock>>>( m_OutputImage,
      m_InputImageSize, m_OutputImageSize, offset );
    cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
    cudaCastToHost( dimGrid.x * dimBlock.x, m_OutputImage, tmp_src, &dst[offset] );
  }
  delete[] tmp_src;
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
template <typename TTextureType>
cudaError_t
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaBindTextureToArray( cudaArray* dst, const TInternalImageType* src,
  cudaExtent& extent, TTextureType& tex, cudaChannelFormatDesc& desc,
  bool normalized, bool onDevice )
{
  cudaMemcpy3DParms copyParams = {0};
  copyParams.extent   = extent;
  copyParams.kind   = onDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  copyParams.dstArray = dst;
  copyParams.srcPtr   = make_cudaPitchedPtr(
    const_cast<TInternalImageType*>(src),
    extent.width * sizeof(TInternalImageType), extent.width, extent.height );
  cuda::cudaMemcpy3D( &copyParams );

  tex.normalized   = normalized;
  tex.filterMode   = cudaFilterMode;
  tex.addressMode[0] = tex.normalized ? cudaAddressModeMirror: cudaAddressModeClamp;
  tex.addressMode[1] = tex.normalized ? cudaAddressModeMirror: cudaAddressModeClamp;
  tex.addressMode[2] = tex.normalized ? cudaAddressModeMirror: cudaAddressModeClamp;
  return cuda::cudaBindTextureToArray( tex, dst, desc );
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToHost( size_t size, const TInternalImageType* src,
  TInternalImageType* tmp_src, TImageType* dst )
{
  cuda::cudaMemcpy( tmp_src, src, size, cudaMemcpyDeviceToHost );
  for ( size_t i = 0; i != size; ++i )
  {
    dst[ i ] = static_cast<TImageType>( tmp_src[i] );
  }
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToHost( int3 size, const TInternalImageType* src, TImageType* dst )
{
  cudaExtent volumeExtent = make_cudaExtent(size.x, size.y, size.z);
  cudaCastToType<TInternalImageType, TImageType>(
    volumeExtent, src, dst, cudaMemcpyDeviceToHost, m_CastOnGPU );
}

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToDevice( int3 size, const TImageType* src, TInternalImageType* dst )
{
  cudaExtent volumeExtent = make_cudaExtent( size.x, size.y, size.z );
  cudaCastToType<TImageType, TInternalImageType>(
    volumeExtent, src, dst, cudaMemcpyHostToDevice, m_CastOnGPU );
}

/* check for double TInputImageType or TOutputImageType */
template <class T> inline bool is_double();
template <class T> inline bool is_double() {return false;}
template <       > inline bool is_double<double>() {return true;}

template <>
float* cuda::cudaCastToType<float, float>( cudaExtent& volumeExtent,
  const float* src, float* dst, cudaMemcpyKind direction, bool UseGPU )
{
  const size_t voxelsPerSlice = volumeExtent.width * volumeExtent.height;
  cuda::cudaMemcpy( dst, src, voxelsPerSlice * volumeExtent.depth, direction );
  return dst;
}

template <class TInputImageType, class TOutputImageType>
TOutputImageType* cuda::cudaCastToType( cudaExtent& volumeExtent,
  const TInputImageType* src, TOutputImageType* dst, cudaMemcpyKind direction, bool UseGPU )
{
  cudaDeviceProp prop;
  size_t offset = 0;
  const size_t voxelsPerSlice = volumeExtent.width * volumeExtent.height;

  // std::max( size_t, size_t ) does not exist
  dim3 dimBlock( std::min( static_cast<int>(
    std::max( (long long)volumeExtent.width, (long long)volumeExtent.height ) ), 512 ) );
  dim3 dimGrid( (unsigned int)( voxelsPerSlice / dimBlock.x ) );

  /* Not a perfect fit, fix it */
  if ( dimBlock.x * dimGrid.x != voxelsPerSlice ) ++dimGrid.x;

  clock_t start = clock();

  /* only devices from compute capability 1.3 support double precision on the device */
  cuda::cudaGetDeviceProperties(&prop, 0);
  bool device_less_2_0 = (prop.major == 1 && prop.minor < 3);

  switch ( direction )
  {
  case cudaMemcpyHostToDevice:
    if ( is_double<TOutputImageType>() && device_less_2_0 )
    {
      //throw itk::ExceptionObject( "GPU doesn't support double-precision" );
      throw std::exception();
    }

    if ( !UseGPU )
    {
      size_t nof_elements = voxelsPerSlice * volumeExtent.depth;

      /* Allocate memory on host, copy over data (and cast) and copy results to GPU. */
      TOutputImageType* tmp = new TOutputImageType[ nof_elements ];
      for ( size_t i = 0; i != nof_elements; ++i )
      {
        tmp[ i ] = static_cast<TOutputImageType>( src[ i ] );
      }
      cuda::cudaMemcpy( dst, tmp, nof_elements, cudaMemcpyHostToDevice );
    }
    else
    {
      TInputImageType* tmp = cuda::cudaMalloc<TInputImageType>( voxelsPerSlice );

      /* Process each slice separately, copy source to GPU, and cast/copy in kernel. */
      for ( int slice = 0; slice != volumeExtent.depth; ++slice, offset += voxelsPerSlice )
      {
        cuda::cudaMemcpy( tmp, src + offset, voxelsPerSlice, cudaMemcpyHostToDevice );
        cast_to_type<TInputImageType, TOutputImageType><<<dimGrid, dimBlock>>>(
          dst + offset, tmp, voxelsPerSlice );
        cuda::cudaCheckMsg( "kernel launch failed: cast_to_type" );
      }
      cudaFree( tmp );
    }
    break;
  case cudaMemcpyDeviceToHost:
    if ( is_double<TInputImageType>() && device_less_2_0 )
    {
      //throw itk::ExceptionObject( "GPU doesn't support double-precision" );
      throw std::exception();
    }

    if ( !UseGPU )
    {
      size_t nof_elements = voxelsPerSlice * volumeExtent.depth;

      /* Allocate memory on host, copy data from GPU and cast. */
      TInputImageType* tmp = new TInputImageType[ nof_elements ];
      cuda::cudaMemcpy( tmp, src, nof_elements, cudaMemcpyDeviceToHost );
      for ( size_t i = 0; i != nof_elements; ++i )
      {
        dst[ i ] = static_cast<TOutputImageType>( tmp[ i ] );
      }
    }
    else
    {
      TOutputImageType* tmp = cuda::cudaMalloc<TOutputImageType>( voxelsPerSlice );

      /* Process each slice separately, cast/copy in kernel and copy results to host. */
      for ( int slice = 0; slice != volumeExtent.depth; ++slice, offset += voxelsPerSlice )
      {
        cast_to_type<TInputImageType, TOutputImageType><<<dimGrid, dimBlock>>>(
          tmp, src + offset, voxelsPerSlice );
        cuda::cudaCheckMsg( "kernel launch failed: cast_to_type" );
        cuda::cudaMemcpy( dst + offset, tmp, voxelsPerSlice, cudaMemcpyDeviceToHost );
      }
      cudaFree( tmp );
    }
    break;
  }

//   std::cout << "type conversion took " << clock() - start << "ms for "
//     << voxelsPerSlice * volumeExtent.depth << " elements" << std::endl;
  return dst;
}


/** Template linker errors...
 * http://www.parashift.com/c++-faq-lite/templates.html#faq-35.14
 * Note that gcc requires these lines at the bottom of this file.
 */
template class cuda::CUDAResampleImageFilter<double, short, float>;
template class cuda::CUDAResampleImageFilter<double, int  , float>;
template class cuda::CUDAResampleImageFilter<double, float, float>;

