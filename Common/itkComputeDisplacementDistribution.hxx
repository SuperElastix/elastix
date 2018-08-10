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
#ifndef __itkComputeDisplacementDistribution_hxx
#define __itkComputeDisplacementDistribution_hxx

#include "itkComputeDisplacementDistribution.h"

#include <string>
#include "vnl/vnl_math.h"
#include "vnl/vnl_fastops.h"
#include "vnl/vnl_diag_matrix.h"

#include "itkImageScanlineIterator.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "itkCropImageFilter.h"
#include "itkMirrorPadImageFilter.h"
#include "itkZeroFluxNeumannPadImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

namespace itk
{

/**
 * ************************* Constructor ************************
 */

template< class TFixedImage, class TTransform >
ComputeDisplacementDistribution< TFixedImage, TTransform >
::ComputeDisplacementDistribution()
{
  this->m_FixedImage                   = NULL;
  this->m_FixedImageMask               = NULL;
  this->m_Transform                    = NULL;
  this->m_FixedImageMask               = NULL;
  this->m_NumberOfJacobianMeasurements = 0;
  this->m_SampleContainer              = 0;

  /** Threading related variables. */
  this->m_UseMultiThread = true;
  this->m_Threader       = ThreaderType::New();
  this->m_Threader->SetUseThreadPool( false );

  /** Initialize the m_ThreaderParameters. */
  this->m_ThreaderParameters.st_Self = this;

  // Multi-threading structs
  this->m_ComputePerThreadVariables     = NULL;
  this->m_ComputePerThreadVariablesSize = 0;

} // end Constructor


/**
 * ************************* Destructor ************************
 */

template< class TFixedImage, class TTransform >
ComputeDisplacementDistribution< TFixedImage, TTransform >
::~ComputeDisplacementDistribution()
{
  delete[] this->m_ComputePerThreadVariables;
} // end Destructor


/**
 * ************************* InitializeThreadingParameters ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::InitializeThreadingParameters( void )
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */
  const ThreadIdType numberOfThreads = this->m_Threader->GetNumberOfThreads();

  /** Only resize the array of structs when needed. */
  if( this->m_ComputePerThreadVariablesSize != numberOfThreads )
  {
    delete[] this->m_ComputePerThreadVariables;
    this->m_ComputePerThreadVariables     = new AlignedComputePerThreadStruct[ numberOfThreads ];
    this->m_ComputePerThreadVariablesSize = numberOfThreads;
  }

  /** Some initialization. */
  for( ThreadIdType i = 0; i < numberOfThreads; ++i )
  {
    this->m_ComputePerThreadVariables[ i ].st_MaxJJ                 = NumericTraits< double >::Zero;
    this->m_ComputePerThreadVariables[ i ].st_Displacement          = NumericTraits< double >::Zero;
    this->m_ComputePerThreadVariables[ i ].st_DisplacementSquared   = NumericTraits< double >::Zero;
    this->m_ComputePerThreadVariables[ i ].st_NumberOfPixelsCounted = NumericTraits< SizeValueType >::Zero;
  }

} // end InitializeThreadingParameters()


/**
 * ************************* ComputeSingleThreaded ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::ComputeSingleThreaded( const ParametersType & mu,
  double & jacg, double & maxJJ, std::string methods )
{
  /** This function computes four terms needed for the automatic parameter
   * estimation using voxel displacement distribution estimation method.
   * The equation number refers to the SPIE paper.
   * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
   */

  /** Initialize. */
  maxJJ = jacg = 0.0;

  /** Get samples. */
  ImageSampleContainerPointer sampleContainer = 0;
  this->SampleFixedImageForJacobianTerms( sampleContainer );
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get the number of parameters. */
  const unsigned int P = static_cast< unsigned int >(
    this->m_Transform->GetNumberOfParameters() );

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  this->m_ScaledCostFunction->SetScales( scales );

  /** Get the exact gradient. */
  this->m_ExactGradient = DerivativeType( P );
  this->m_ExactGradient.Fill( 0.0 );
  this->GetScaledDerivative( mu, this->m_ExactGradient );

  /** Get transform and set current position. */
  const unsigned int outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end   = sampleContainer->End();
  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind
    = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType jacj( outdim, sizejacind );
  jacj.Fill( 0.0 );
  NonZeroJacobianIndicesType jacind( sizejacind );
  jacind[ 0 ] = 0;
  if( sizejacind > 1 ) { jacind[ 1 ] = 0; }

  /**
   * Compute maxJJ and jac*gradient
   */
  DerivativeType Jgg( outdim );
  Jgg.Fill( 0.0 );
  std::vector< double > JGG_k;
  double                globalDeformation = 0.0;
  const double          sqrt2             = vcl_sqrt( static_cast< double >( 2.0 ) );
  JacobianType          jacjjacj( outdim, outdim );

  samplenr = 0;
  for( iter = begin; iter != end; ++iter )
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = ( *iter ).Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian( point, jacj, jacind );

    /** Apply scales, if necessary. */
    if( this->GetUseScales() )
    {
      for( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[ pi ];
        jacj.scale_column( pi, 1.0 / scales[ p ] );
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math_sqr( jacj.frobenius_norm() );

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt( jacjjacj, jacj, jacj );
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = vnl_math_max( maxJJ, JJ_j );

    /** Compute the matrix of jac*gradient */
    for( unsigned int i = 0; i < outdim; ++i )
    {
      double temp = 0.0;
      for( unsigned int j = 0; j < sizejacind; ++j )
      {
        int pj = jacind[ j ];
        temp += jacj( i, j ) * this->m_ExactGradient( pj );
      }
      Jgg( i ) = temp;
    }

    globalDeformation += Jgg.magnitude();
    JGG_k.push_back( Jgg.magnitude() );
    ++samplenr;

  } // end loop over sample container

  if( methods == "95percentile" )
  {
    /** Compute the 95% percentile of the distribution of JGG_k */
    unsigned int d = static_cast< unsigned int >( nrofsamples * 0.95 );
    std::sort( JGG_k.begin(), JGG_k.end() );
    jacg = ( JGG_k[ d - 1 ] + JGG_k[ d ] + JGG_k[ d + 1 ] ) / 3.0;
  }
  else if( methods == "2sigma" )
  {
    /** Compute the sigma of the distribution of JGG_k. */
    double sigma    = 0.0;
    double mean_JGG = globalDeformation / samplenr;
    for( unsigned int i = 0; i < nrofsamples; ++i )
    {
      sigma += vnl_math_sqr( JGG_k[ i ] - mean_JGG );
    }
    sigma /= ( nrofsamples - 1 ); // unbiased estimation
    jacg   = mean_JGG + 2.0 * vcl_sqrt( sigma );
  }

} // end ComputeSingleThreaded()


/**
 * ************************* Compute ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::Compute( const ParametersType & mu,
  double & jacg, double & maxJJ, std::string methods )
{
  /** Option for now to still use the single threaded code. */
  if( !this->m_UseMultiThread )
  {
    return this->ComputeSingleThreaded( mu, jacg, maxJJ, methods );
  }
  // The multi-threaded route only supports methods == 2sigma for now

  /** Initialize multi-threading. */
  this->InitializeThreadingParameters();

  /** Tackle stuff needed before multi-threading. */
  this->BeforeThreadedCompute( mu );

  /** Launch multi-threaded computation. */
  this->LaunchComputeThreaderCallback();

  /** Gather the jacg, maxJJ values from all threads. */
  this->AfterThreadedCompute( jacg, maxJJ );

} // end Compute()


/**
 * *********************** BeforeThreadedCompute***************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::BeforeThreadedCompute( const ParametersType & mu )
{
  /** Get the number of parameters. */
  this->m_NumberOfParameters = static_cast< unsigned int >(
    this->m_Transform->GetNumberOfParameters() ); // why is this parameter needed?

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  this->m_ScaledCostFunction->SetScales( scales );

  /** Get the exact gradient. */
  this->m_ExactGradient = DerivativeType( this->m_NumberOfParameters );
  this->m_ExactGradient.Fill( 0.0 );
  this->GetScaledDerivative( mu, this->m_ExactGradient );

  /** Get samples. */
  this->SampleFixedImageForJacobianTerms( this->m_SampleContainer );

} // end BeforeThreadedCompute()


/**
 * *********************** LaunchComputeThreaderCallback***************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::LaunchComputeThreaderCallback( void ) const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod( this->ComputeThreaderCallback,
    const_cast< void * >( static_cast< const void * >( &this->m_ThreaderParameters ) ) );

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchComputeThreaderCallback()


/**
 * ************ ComputeThreaderCallback ****************************
 */

template< class TFixedImage, class TTransform >
ITK_THREAD_RETURN_TYPE
ComputeDisplacementDistribution< TFixedImage, TTransform >
::ComputeThreaderCallback( void * arg )
{
  /** Get the current thread id and user data. */
  ThreadInfoType *             infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType                 threadID   = infoStruct->ThreadID;
  MultiThreaderParameterType * temp
    = static_cast< MultiThreaderParameterType * >( infoStruct->UserData );

  /** Call the real implementation. */
  temp->st_Self->ThreadedCompute( threadID );

  return ITK_THREAD_RETURN_VALUE;

} // end ComputeThreaderCallback()


/**
 * ************************* ThreadedCompute ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::ThreadedCompute( ThreadIdType threadId )
{
  /** Get sample container size, number of threads, and output space dimension. */
  const SizeValueType sampleContainerSize = this->m_SampleContainer->Size();
  const ThreadIdType  numberOfThreads     = this->m_Threader->GetNumberOfThreads();
  const unsigned int  outdim              = this->m_Transform->GetOutputSpaceDimension();

  /** Get a handle to the scales vector */
  const ScalesType & scales = this->GetScales();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast< unsigned long >( vcl_ceil( static_cast< double >( sampleContainerSize )
    / static_cast< double >( numberOfThreads ) ) );

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end   = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_begin = ( pos_begin > sampleContainerSize ) ? sampleContainerSize : pos_begin;
  pos_end   = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind
    = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType jacj( outdim, sizejacind );
  jacj.Fill( 0.0 );
  NonZeroJacobianIndicesType jacind( sizejacind );
  jacind[ 0 ] = 0;
  if( sizejacind > 1 ) { jacind[ 1 ] = 0; }

  /** Temporaries. */
  //std::vector< double > JGG_k; not here so only mean + 2 sigma is supported
  DerivativeType Jgg( outdim ); Jgg.Fill( 0.0 );
  const double   sqrt2 = vcl_sqrt( static_cast< double >( 2.0 ) );
  JacobianType   jacjjacj( outdim, outdim );
  double         maxJJ                 = 0.0;
  double         jggMagnitude          = 0.0;
  double         displacement          = 0.0;
  double         displacementSquared   = 0.0;
  unsigned long  numberOfPixelsCounted = 0;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = this->m_SampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend   = this->m_SampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend   += (int)pos_end;

  /** Loop over the fixed image to calculate the mean squares. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = ( *threader_fiter ).Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian( point, jacj, jacind );

    /** Apply scales, if necessary. */
    if( this->GetUseScales() )
    {
      for( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[ pi ];
        jacj.scale_column( pi, 1.0 / scales[ p ] );
      }
    }

    /** Compute 1st part of JJ: ||J_j||_F^2. */
    double JJ_j = vnl_math_sqr( jacj.frobenius_norm() );

    /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
    vnl_fastops::ABt( jacjjacj, jacj, jacj ); // is this thread-safe?
    JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

    /** Max_j [JJ_j]. */
    maxJJ = vnl_math_max( maxJJ, JJ_j );

    /** Compute the displacement  jac * gradient. */
    for( unsigned int i = 0; i < outdim; ++i )
    {
      double temp = 0.0;
      for( unsigned int j = 0; j < sizejacind; ++j )
      {
        int pj = jacind[ j ];
        temp += jacj( i, j ) * this->m_ExactGradient( pj );
      }
      Jgg( i ) = temp;
    }

    /** Sum the Jgg displacement for later use. */
    jggMagnitude         = Jgg.magnitude();
    displacement        += jggMagnitude;
    displacementSquared += vnl_math_sqr( jggMagnitude );
    numberOfPixelsCounted++;
  }

  /** Update the thread struct once. */
  this->m_ComputePerThreadVariables[ threadId ].st_MaxJJ                 = maxJJ;
  this->m_ComputePerThreadVariables[ threadId ].st_Displacement          = displacement;
  this->m_ComputePerThreadVariables[ threadId ].st_DisplacementSquared   = displacementSquared;
  this->m_ComputePerThreadVariables[ threadId ].st_NumberOfPixelsCounted = numberOfPixelsCounted;

} // end ThreadedCompute()


/**
 * *********************** AfterThreadedCompute***************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::AfterThreadedCompute( double & jacg, double & maxJJ )
{
  const ThreadIdType numberOfThreads = this->m_Threader->GetNumberOfThreads();

  /** Reset all variables. */
  maxJJ = 0.0;
  double displacement        = 0.0;
  double displacementSquared = 0.0;
  this->m_NumberOfPixelsCounted = 0.0;

  /** Accumulate thread results. */
  for( ThreadIdType i = 0; i < numberOfThreads; ++i )
  {
    maxJJ                          = vnl_math_max( maxJJ, this->m_ComputePerThreadVariables[ i ].st_MaxJJ );
    displacement                  += this->m_ComputePerThreadVariables[ i ].st_Displacement;
    displacementSquared           += this->m_ComputePerThreadVariables[ i ].st_DisplacementSquared;
    this->m_NumberOfPixelsCounted += this->m_ComputePerThreadVariables[ i ].st_NumberOfPixelsCounted;

    /** Reset all variables for the next resolution. */
    this->m_ComputePerThreadVariables[ i ].st_MaxJJ                 = 0;
    this->m_ComputePerThreadVariables[ i ].st_Displacement          = 0;
    this->m_ComputePerThreadVariables[ i ].st_DisplacementSquared   = 0;
    this->m_ComputePerThreadVariables[ i ].st_NumberOfPixelsCounted = 0;
  }

  /** Compute the sigma of the distribution of the displacements. */
  const double meanDisplacement = displacement / this->m_NumberOfPixelsCounted;
  const double sigma            = displacementSquared / this->m_NumberOfPixelsCounted - vnl_math_sqr( meanDisplacement );

  jacg = meanDisplacement + 2.0 * vcl_sqrt( sigma );

} // end AfterThreadedCompute()


/**
 * ************************* ComputeUsingSearchDirection ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::ComputeUsingSearchDirection( const ParametersType & mu,
  double & jacg, double & maxJJ, std::string methods )
{
  /** This function computes four terms needed for the automatic parameter
   * estimation using voxel displacement distribution estimation method.
   * The equation number refers to the SPIE paper.
   * Term 1: jacg = mean( J_j * g ) + var( J_j * g ).
   */

  /** Initialize. */
  maxJJ = jacg = 0.0;

  /** Get samples. */
  ImageSampleContainerPointer sampleContainer = 0;
  this->SampleFixedImageForJacobianTerms( sampleContainer );
  const SizeValueType nrofsamples = sampleContainer->Size();

  /** Get the number of parameters. */
  const unsigned int P = static_cast< unsigned int >(
    this->m_Transform->GetNumberOfParameters() );

  /** Get scales vector */
  const ScalesType & scales = this->GetScales();
  this->m_ScaledCostFunction->SetScales( scales );

  /** Get the exact gradient. */
  DerivativeType exactgradient( P );
  exactgradient = mu;

  /** Get transform and set current position. */
  typename TransformType::Pointer transform = this->m_Transform;
  const unsigned int outdim = this->m_Transform->GetOutputSpaceDimension();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator iter;
  typename ImageSampleContainerType::ConstIterator begin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator end   = sampleContainer->End();
  unsigned int samplenr = 0;

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const SizeValueType sizejacind
    = this->m_Transform->GetNumberOfNonZeroJacobianIndices();
  JacobianType jacj( outdim, sizejacind );
  jacj.Fill( 0.0 );
  NonZeroJacobianIndicesType jacind( sizejacind );
  jacind[ 0 ] = 0;
  if( sizejacind > 1 ) { jacind[ 1 ] = 0; }

  /**
   * Compute maxJJ and jac*gradient
   */
  DerivativeType Jgg( outdim );
  Jgg.Fill( 0.0 );
  std::vector< double > JGG_k;
  double                globalDeformation = 0.0;
  JacobianType          jacjjacj( outdim, outdim );

  samplenr = 0;
  for( iter = begin; iter != end; ++iter )
  {
    /** Read fixed coordinates and get Jacobian. */
    const FixedImagePointType & point = ( *iter ).Value().m_ImageCoordinates;
    this->m_Transform->GetJacobian( point, jacj, jacind );

    /** Apply scales, if necessary. */
    if( this->GetUseScales() )
    {
      for( unsigned int pi = 0; pi < sizejacind; ++pi )
      {
        const unsigned int p = jacind[ pi ];
        jacj.scale_column( pi, 1.0 / scales[ p ] );
      }
    }

    /** Compute the matrix of jac*gradient */
    for( unsigned int i = 0; i < outdim; ++i )
    {
      double temp = 0.0;
      for( unsigned int j = 0; j < sizejacind; ++j )
      {
        int pj = jacind[ j ];
        temp += jacj( i, j ) * exactgradient( pj );
      }
      Jgg( i ) = temp;
    }

    globalDeformation += Jgg.magnitude();
    JGG_k.push_back( Jgg.magnitude() );
    ++samplenr;

  } // end loop over sample container

  if( methods == "95percentile" )
  {
    /** Compute the 95% percentile of the distribution of JGG_k */
    unsigned int d = static_cast< unsigned int >( nrofsamples * 0.95 );
    std::sort( JGG_k.begin(), JGG_k.end() );
    jacg = ( JGG_k[ d - 1 ] + JGG_k[ d ] + JGG_k[ d + 1 ] ) / 3.0;
  }
  else if( methods == "2sigma" )
  {
    /** Compute the sigma of the distribution of JGG_k. */
    double sigma    = 0.0;
    double mean_JGG = globalDeformation / samplenr;
    for( unsigned int i = 0; i < nrofsamples; ++i )
    {
      sigma += vnl_math_sqr( JGG_k[ i ] - mean_JGG );
    }
    sigma /= ( nrofsamples - 1 ); // unbiased estimation
    jacg   = mean_JGG + 2.0 * vcl_sqrt( sigma );
  }
} // end ComputeUsingSearchDirection()


/**
 * ************************* SampleFixedImageForJacobianTerms ************************
 */

template< class TFixedImage, class TTransform >
void
ComputeDisplacementDistribution< TFixedImage, TTransform >
::SampleFixedImageForJacobianTerms(
  ImageSampleContainerPointer & sampleContainer )
{
  /** Set up grid sampler. */
  ImageGridSamplerPointer sampler = ImageGridSamplerType::New();
  //  ImageFullSamplerPointer sampler = ImageFullSamplerType::New();
  sampler->SetInput( this->m_FixedImage );
  sampler->SetInputImageRegion( this->GetFixedImageRegion() );
  sampler->SetMask( this->m_FixedImageMask );

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  SizeValueType nrofsamples = this->m_NumberOfJacobianMeasurements;
  sampler->SetNumberOfSamples( nrofsamples );

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  sampleContainer = sampler->GetOutput();
  nrofsamples     = sampleContainer->Size();

  if( nrofsamples == 0 )
  {
    itkExceptionMacro(
      << "No valid voxels (0/" << this->m_NumberOfJacobianMeasurements
      << ") found to estimate the AdaptiveStochasticGradientDescent parameters." );
  }
} // end SampleFixedImageForJacobianTerms()


} // end namespace itk

#endif // end #ifndef __itkComputeDisplacementDistribution_hxx
