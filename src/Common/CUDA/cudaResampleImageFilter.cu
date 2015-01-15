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


/**
 * ******************* Constructor ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::CUDAResampleImageFilter()
  : m_CoeffsX( NULL )
  , m_CoeffsY( NULL )
  , m_CoeffsZ( NULL )
  , m_InputImage( NULL )
  , m_InputImageSize( make_uint3( 0, 0, 0 ) )
  , m_Device( 0 )
  , m_MaxNumberOfVoxelsPerIteration( 1 << 20 )
{
  this->m_CastOnGPU = false;
  this->m_UseFastCUDAKernel = false;

} // end Constructor


/**
 * ******************* Destructor ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::~CUDAResampleImageFilter()
{
  this->cudaUnInit();
} // end Destructor


/**
 * ******************* cudaInit ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaInit( void )
{
  this->checkExecutionParameters();
  cuda::cudaSetDevice( this->m_Device ); // always 0?
  this->m_ChannelDescCoeff = cudaCreateChannelDesc<TInternalImageType>();

} // end cudaInit()


/**
 * ******************* cudaUnInit ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaUnInit( void )
{
  cuda::cudaUnbindTexture( m_tex_coeffsX );
  cuda::cudaUnbindTexture( m_tex_coeffsY );
  cuda::cudaUnbindTexture( m_tex_coeffsZ );
  cuda::cudaUnbindTexture( m_tex_inputImage );
  cuda::cudaFreeArray( this->m_CoeffsX );
  cuda::cudaFreeArray( this->m_CoeffsY );
  cuda::cudaFreeArray( this->m_CoeffsZ );
  cuda::cudaFreeArray( this->m_InputImage );
  cuda::cudaFree( this->m_OutputImage );

} // end cudaUnInit()


/**
 * ******************* checkExecutionParameters ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
int
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::checkExecutionParameters( void )
{
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount( &deviceCount );
  return ( err == cudaSuccess ) ? ( deviceCount == 0 ) : 1;

} // end checkExecutionParameters()


/**
 * ******************* cudaCopyImageSymbols ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCopyImageSymbols(
  const float3 & inputImageSpacing,  const float3 & inputImageOrigin,
  const float3 & outputImageSpacing, const float3 & outputImageOrigin,
  const float defaultPixelValue )
{
  /* Copy some constant parameters to the GPU's constant cache. */
  cuda::cudaMemcpyToSymbol( CUInputImageSpacing,  inputImageSpacing,
    cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUInputImageOrigin,   inputImageOrigin,
    cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUOutputImageSpacing, outputImageSpacing,
    cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUOutputImageOrigin,  outputImageOrigin,
    cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUDefaultPixelValue,  defaultPixelValue,
    cudaMemcpyHostToDevice );

} // end cudaCopyImageSymbols()


/**
 * ******************* cudaCopyGridSymbols ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCopyGridSymbols( const float3 & gridSpacing,
  const float3 & gridOrigin, const uint3 & gridSize )
{
  /* Copy some constant parameters to the GPU's constant cache. */
  cuda::cudaMemcpyToSymbol( CUGridSpacing, gridSpacing, cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUGridOrigin,  gridOrigin,  cudaMemcpyHostToDevice );
  cuda::cudaMemcpyToSymbol( CUGridSize,    gridSize,    cudaMemcpyHostToDevice );

} // end cudaCopyGridSymbols()


/**
 * ******************* cudaMallocTransformationData ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaMallocTransformationData( const uint3 & gridSize,
  const TInterpolatorPrecisionType* params )
{
  const unsigned int nrOfParametersPerDimension = gridSize.x * gridSize.y * gridSize.z;
  cudaExtent gridExtent = make_cudaExtent( gridSize.x, gridSize.y, gridSize.z );

  /* Allocate memory on the GPU for the interpolation texture. */
  cuda::cudaMalloc3DArray( &this->m_CoeffsX, &this->m_ChannelDescCoeff, gridExtent );
  cuda::cudaMalloc3DArray( &this->m_CoeffsY, &this->m_ChannelDescCoeff, gridExtent );
  cuda::cudaMalloc3DArray( &this->m_CoeffsZ, &this->m_ChannelDescCoeff, gridExtent );

  /* Convert TInterpolatorPrecisionType to float, only thing textures support. */
#if 1
  //clock_t start = clock();
  TInternalImageType* params_tmp = new TInternalImageType[ nrOfParametersPerDimension * 3 ];
  for ( size_t i = 0; i != nrOfParametersPerDimension * 3; ++i )
  {
    params_tmp[ i ] = static_cast<TInternalImageType>( params[ i ] );
  }
  //std::cout << "parameter type conversion took "
  //  << clock() - start << "ms for "
  //  << nrOfParametersPerDimension * 3 << " elements" << std::endl;
  cudaBindTextureToArray( m_CoeffsX, &params_tmp[ 0 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsX, this->m_ChannelDescCoeff );
  cudaBindTextureToArray( m_CoeffsY, &params_tmp[ 1 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsY, this->m_ChannelDescCoeff );
  cudaBindTextureToArray( m_CoeffsZ, &params_tmp[ 2 * nrOfParametersPerDimension ],
    gridExtent, m_tex_coeffsZ, this->m_ChannelDescCoeff );
  delete[] params_tmp;
#else
  /* There are some problems with Device2Device copy when src is not a pitched or 3D array. */
  TInternalImageType* params_gpu
    = cuda::cudaMalloc<TInternalImageType>( nrOfParametersPerDimension );

  /* Create the B-spline coefficients texture. */
  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 0 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_CoeffsX, params_gpu, gridExtent, m_tex_coeffsX,
    m_ChannelDescCoeff, false, true );

  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 1 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_CoeffsY, params_gpu, gridExtent, m_tex_coeffsY,
    m_ChannelDescCoeff, false, true );

  cudaCastToType<TInterpolatorPrecisionType, TInternalImageType>(
    gridExtent, &params[ 2 * nrOfParametersPerDimension ],
    params_gpu, cudaMemcpyHostToDevice, m_CastOnGPU );
  cudaBindTextureToArray( m_CoeffsZ, params_gpu, gridExtent, m_tex_coeffsZ,
    m_ChannelDescCoeff, false, true );

  cuda::cudaFree( params_gpu );
#endif

} // end cudaMallocTransformationData()


/**
 * ******************* cudaMallocImageData ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaMallocImageData( const uint3 & inputSize,
  const uint3 & outputSize, const TImageType* data )
{
  this->m_InputImageSize        = inputSize;
  this->m_OutputImageSize       = outputSize;
  this->m_NumberOfInputVoxels   = this->m_InputImageSize.x
    * this->m_InputImageSize.y * this->m_InputImageSize.z;
  size_t nrOfOutputVoxels       = this->m_OutputImageSize.x
    * this->m_OutputImageSize.y * this->m_OutputImageSize.z;
  this->m_MaxNumberOfVoxelsPerIteration = std::min(
    static_cast<unsigned int>( nrOfOutputVoxels ),
    this->m_MaxNumberOfVoxelsPerIteration );

  cudaExtent volumeExtent = make_cudaExtent(
    this->m_InputImageSize.x, this->m_InputImageSize.y, this->m_InputImageSize.z );

  /* Allocate in memory and PreFilter image. We need to cast to float if not
   * already, because linear filtering only works with floating point values.
   * NOTE: the input image needs to be allocated on the GPU entirely,
   * which may fail for large images and low-end GPU's.
   */
  TInternalImageType* inputImage
    = cuda::cudaMalloc<TInternalImageType>( this->m_NumberOfInputVoxels );
  cudaCastToDevice( this->m_InputImageSize, data, inputImage );
  /** Prefiltering is performed in-place. */
  CubicBSplinePrefilter3D( inputImage,
    volumeExtent.width, volumeExtent.height, volumeExtent.depth );

  /* XXX - cudaMemcpy3D fails if a DeviceToDevice copy src is not allocated
   * with cudaMallocPitch or cudaMalloc3D, so we need this hack to get the data there.
   */
  TInternalImageType* tmpImage
    = new TInternalImageType[ this->m_NumberOfInputVoxels ];
  cuda::cudaMemcpy( tmpImage, inputImage,
    this->m_NumberOfInputVoxels, cudaMemcpyDeviceToHost );
  cuda::cudaFree( inputImage );

  /* Create the image interpolation texture. */
  cuda::cudaMalloc3DArray( &this->m_InputImage, &this->m_ChannelDescCoeff, volumeExtent );
  cudaBindTextureToArray( this->m_InputImage, tmpImage,
    volumeExtent, m_tex_inputImage, this->m_ChannelDescCoeff );
  delete[] tmpImage;

  /* Allocate destination array. */
  this->m_OutputImage = cuda::cudaMalloc<TInternalImageType>(
    this->m_MaxNumberOfVoxelsPerIteration );

} // end cudaMallocImageData()


/**
 * ******************* GenerateData ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::GenerateData( TImageType* dst )
{
  /* Split up applying the transformation due to memory constraints and make
   * sure we never overflow the output image dimensions.
   */
  const size_t nrOfOutputVoxels = this->m_OutputImageSize.x
    * this->m_OutputImageSize.y * this->m_OutputImageSize.z;
  dim3 dimBlock( 256 );
  dim3 dimGrid( this->m_MaxNumberOfVoxelsPerIteration / dimBlock.x );
  size_t offset = 0;

  TInternalImageType* tmp_src = new TInternalImageType[ this->m_MaxNumberOfVoxelsPerIteration ];
  if ( nrOfOutputVoxels > this->m_MaxNumberOfVoxelsPerIteration )
  {
    /* Do a full run of m_MaxnrOfVoxelsPerIteration voxels. */
    for ( offset = 0; offset <= nrOfOutputVoxels - this->m_MaxNumberOfVoxelsPerIteration;
      offset += this->m_MaxNumberOfVoxelsPerIteration )
    {
      resample_image<<<dimGrid, dimBlock>>>( this->m_OutputImage,
        this->m_InputImageSize, this->m_OutputImageSize, offset, this->m_UseFastCUDAKernel );
      cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
      cudaCastToHost( this->m_MaxNumberOfVoxelsPerIteration,
        this->m_OutputImage, tmp_src, &dst[ offset ] );
    }
  }

  /* Do the remainder ensuring again dimGrid * dimBlock is less than image size. */
  dimGrid = dim3((unsigned int)(nrOfOutputVoxels - offset)) / dimBlock;
  resample_image<<<dimGrid, dimBlock>>>( this->m_OutputImage,
    this->m_InputImageSize, this->m_OutputImageSize, offset, this->m_UseFastCUDAKernel );
  cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
  cudaCastToHost( dimGrid.x * dimBlock.x, m_OutputImage, tmp_src, &dst[ offset ] );

  /* Do the final amount of voxels < dimBlock. */
  offset += dimGrid.x * dimBlock.x;
  dimBlock = dim3((unsigned int)(nrOfOutputVoxels - offset));
  dimGrid  = dim3( 1 );

  if ( dimBlock.x > 0 )
  {
    resample_image<<<dimGrid, dimBlock>>>( this->m_OutputImage,
      this->m_InputImageSize, this->m_OutputImageSize, offset, this->m_UseFastCUDAKernel );
    cuda::cudaCheckMsg( "kernel launch failed: resample_image" );
    cudaCastToHost( dimGrid.x * dimBlock.x, m_OutputImage, tmp_src, &dst[ offset ] );
  }
  delete[] tmp_src;

} // end GenerateData()


/**
 * ******************* cudaBindTextureToArray ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
template <typename TTextureType>
cudaError_t
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaBindTextureToArray( cudaArray* dst, const TInternalImageType* src,
  const cudaExtent & extent, TTextureType& tex, cudaChannelFormatDesc& desc,
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

} // end cudaBindTextureToArray()


/**
 * ******************* cudaCastToHost ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToHost( const size_t sizevalue, const TInternalImageType* src,
  TInternalImageType* tmp_src, TImageType* dst )
{
  cuda::cudaMemcpy( tmp_src, src, sizevalue, cudaMemcpyDeviceToHost );
  for ( size_t i = 0; i != sizevalue; ++i )
  {
    dst[ i ] = static_cast<TImageType>( tmp_src[i] );
  }

} // end cudaCastToHost()


/**
 * ******************* cudaCastToHost ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToHost( const uint3 & sizevalue, const TInternalImageType* src, TImageType* dst )
{
  cudaExtent volumeExtent = make_cudaExtent( sizevalue.x, sizevalue.y, sizevalue.z );
  cudaCastToType<TInternalImageType, TImageType>(
    volumeExtent, src, dst, cudaMemcpyDeviceToHost, m_CastOnGPU );

} // end cudaCastToHost()


/**
 * ******************* cudaCastToDevice ***********************
 */

template <typename TInterpolatorPrecisionType, typename TImageType, typename TInternalImageType>
void
cuda::CUDAResampleImageFilter<TInterpolatorPrecisionType, TImageType, TInternalImageType>
::cudaCastToDevice( const uint3 & sizevalue, const TImageType* src, TInternalImageType* dst )
{
  cudaExtent volumeExtent = make_cudaExtent( sizevalue.x, sizevalue.y, sizevalue.z );
  cudaCastToType<TImageType, TInternalImageType>(
    volumeExtent, src, dst, cudaMemcpyHostToDevice, m_CastOnGPU );

} // end cudaCastToDevice()


/**
 * ******************* is_double ***********************
 */

template <class T> inline bool is_double();
template <class T> inline bool is_double() {return false;}
template <       > inline bool is_double<double>() {return true;}


/**
 * ******************* cudaCastToType ***********************
 */

template <>
float* cuda::cudaCastToType<float, float>( const cudaExtent & volumeExtent,
  const float* src, float* dst, cudaMemcpyKind direction, const bool useGPU )
{
  const size_t voxelsPerSlice = volumeExtent.width * volumeExtent.height;
  cuda::cudaMemcpy( dst, src, voxelsPerSlice * volumeExtent.depth, direction );
  return dst;

} // end cudaCastToType()


/**
 * ******************* cudaCastToType ***********************
 */

template <class TInputImageType, class TOutputImageType>
TOutputImageType* cuda
::cudaCastToType( const cudaExtent & volumeExtent,
  const TInputImageType* src, TOutputImageType* dst,
  cudaMemcpyKind direction, bool useGPU )
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

  //clock_t start = clock();

  /* only devices from compute capability 1.3 support double precision on the device */
  cuda::cudaGetDeviceProperties( &prop, 0 );
  bool device_less_2_0 = ( prop.major == 1 && prop.minor < 3 );

  switch ( direction )
  {
  case cudaMemcpyHostToDevice:
    if ( is_double<TOutputImageType>() && device_less_2_0 )
    {
      throw std::string( "GPU doesn't support double-precision" );
    }

    if ( !useGPU )
    {
      const size_t nof_elements = voxelsPerSlice * volumeExtent.depth;

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
      for ( unsigned int slice = 0; slice != volumeExtent.depth; ++slice, offset += voxelsPerSlice )
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
      throw std::string( "GPU doesn't support double-precision" );
    }

    if ( !useGPU )
    {
      const size_t nof_elements = voxelsPerSlice * volumeExtent.depth;

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
      for ( unsigned int slice = 0; slice != volumeExtent.depth; ++slice, offset += voxelsPerSlice )
      {
        cast_to_type<TInputImageType, TOutputImageType><<<dimGrid, dimBlock>>>(
          tmp, src + offset, voxelsPerSlice );
        cuda::cudaCheckMsg( "kernel launch failed: cast_to_type" );
        cuda::cudaMemcpy( dst + offset, tmp, voxelsPerSlice, cudaMemcpyDeviceToHost );
      }
      cudaFree( tmp );
    }
    break;
  case cudaMemcpyHostToHost:
    break;
  case cudaMemcpyDeviceToDevice:
    break;
  case cudaMemcpyDefault:
    break;
  }

  return dst;

} // end cudaCastToType()


/** Template linker errors...
 * http://www.parashift.com/c++-faq-lite/templates.html#faq-35.14
 * Note that gcc requires these lines at the bottom of this file.
 */
template class cuda::CUDAResampleImageFilter<double, short, float>;
template class cuda::CUDAResampleImageFilter<double, int  , float>;
template class cuda::CUDAResampleImageFilter<double, float, float>;

