/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx
#define _itkAdvancedMeanSquaresImageToImageMetric_hxx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

#include <omp.h> // OpenMP
#include <Eigen/Dense> // Eigen
#include <Eigen/Core> // Eigen

//#define FillDerivatives

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::AdvancedMeanSquaresImageToImageMetric()
{
  this->SetUseImageSampler( true );
  this->SetUseFixedImageLimiter( false );
  this->SetUseMovingImageLimiter( false );

  this->m_UseNormalization = false;
  this->m_NormalizationFactor = 1.0;

  /** SelfHessian related variables, experimental feature. */
  this->m_SelfHessianSmoothingSigma = 1.0;
  this->m_NumberOfSamplesForSelfHessian = 100000;

  /** Threading related variables. */
  this->m_UseOpenMP = false;

} // end Constructor


/**
 * ******************* Destructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::~AdvancedMeanSquaresImageToImageMetric()
{
  this->m_ThreaderValues.resize( 0 );
  this->m_ThreaderDerivatives.resize( 0 );
  this->m_ThreaderNumberOfPixelsCounted.resize( 0 );

} // end Destructor


/**
 * ********************* Initialize ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::Initialize(void) throw ( ExceptionObject )
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  if ( this->GetUseNormalization() )
  {
    /** Try to guess a normalization factor. */
    this->ComputeFixedImageExtrema(
      this->GetFixedImage(),
      this->GetFixedImageRegion() );

    this->ComputeMovingImageExtrema(
      this->GetMovingImage(),
      this->GetMovingImage()->GetBufferedRegion() );

    const double diff1 = this->m_FixedImageTrueMax - this->m_MovingImageTrueMin;
    const double diff2 = this->m_MovingImageTrueMax - this->m_FixedImageTrueMin;
    const double maxdiff = vnl_math_max( diff1, diff2 );

    /** We guess that maxdiff/10 is the maximum average difference that will
     * be observed.
     * \todo We may involve the standard derivation of the image into this estimate.
     */
    this->m_NormalizationFactor = 1.0;
    if ( maxdiff > 1e-10 )
    {
      this->m_NormalizationFactor = 100.0 / maxdiff / maxdiff;
    }
  }
  else
  {
    this->m_NormalizationFactor = 1.0;
  }

} // end Initialize()


/**
 * ******************* PrintSelf *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::PrintSelf( std::ostream& os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << "UseNormalization: "
    << this->m_UseNormalization << std::endl;
  os << "SelfHessianSmoothingSigma: "
    << this->m_SelfHessianSmoothingSigma << std::endl;
  os << "NumberOfSamplesForSelfHessian: "
    << this->m_NumberOfSamplesForSelfHessian << std::endl;

} // end PrintSelf()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
typename AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>::MeasureType
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValue( const TransformParametersType & parameters ) const
{
  itkDebugMacro( "GetValue( " << parameters << " ) " );

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, 0 );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<double>( (*fiter).Value().m_ImageValue );

      /** The difference squared. */
      const RealType diff = movingImageValue - fixedImageValue;
      measure += diff * diff;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Update measure value. */
  double normal_sum = 0.0;
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    normal_sum = this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
  }
  measure *= normal_sum;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template < class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType & derivative ) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits< MeasureType >::Zero;
  this->GetValueAndDerivative( parameters, dummyvalue, derivative );

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits< MeasureType >::Zero;
  derivative = DerivativeType( this->GetNumberOfParameters() );
  derivative.Fill( NumericTraits< DerivativeValueType >::Zero );

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        imageJacobian, nzji,
        measure, derivative );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the measure value and derivative. */
  double normal_sum = 0.0;
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    normal_sum = this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
  }
  measure *= normal_sum;
  derivative *= normal_sum;

  /** The return value. */
  value = measure;

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivativeOpenMP *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivativeOpenMP(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Initialize some threading related parameters. */
  this->InitializeThreadingParameters();

  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const unsigned int nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  std::vector<NonZeroJacobianIndicesType> nzji;
  std::vector<DerivativeType> imageJacobian;
  std::vector<TransformJacobianType> jacobian;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; i++ )
  {
    nzji.push_back( NonZeroJacobianIndicesType( nnzji ) );
    imageJacobian.push_back( DerivativeType( nnzji ) );
    jacobian.push_back( TransformJacobianType( FixedImageDimension, nnzji ) );
    //jacobian[i].Fill( 0.0 ); // needed?
  }

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Loop over the fixed image to calculate the mean squares. */
  const int nthreads = static_cast<int>( this->m_NumberOfThreads );
  omp_set_num_threads( nthreads );
  #pragma omp parallel for
  for( int i = 0; i < sampleContainer->Size(); ++i )
  {
    const int threadId = omp_get_thread_num();

    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = sampleContainer->ElementAt( i ).m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if ( sampleOk )
    {
      this->m_ThreaderNumberOfPixelsCounted[ threadId ]++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( sampleContainer->ElementAt( i ).m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian[ threadId ], nzji[ threadId ] );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian[ threadId ], movingImageDerivative, imageJacobian[ threadId ] );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        imageJacobian[ threadId ], nzji[ threadId ],
        this->m_ThreaderValues[ threadId ],
        this->m_ThreaderDerivatives[ threadId ] );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Accumulate pixel count. */
  this->m_NumberOfPixelsCounted = 0;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; i++ )
  {
    this->m_NumberOfPixelsCounted += this->m_ThreaderNumberOfPixelsCounted[ i ];
  }

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the measure value and derivative. */
  double normal_sum = this->m_NormalizationFactor /
    static_cast<double>( this->m_NumberOfPixelsCounted );

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; i++ )
  {
    value += this->m_ThreaderValues[ i ];
  }
  value *= normal_sum;

  /** Accumulate derivatives. */
  const unsigned int spaceDimension = this->GetNumberOfParameters();
  #pragma omp parallel for
  for( int j = 0; j < spaceDimension; ++j )
  {
    DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
      tmp += this->m_ThreaderDerivatives[ i ][ j ];
    }
    derivative[ j ] = tmp * normal_sum;
  }

} // end GetValueAndDerivativeOpenMP()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Option for now to still use the single threaded code. */
  if ( !this->m_UseMultiThread )
  {
    return this->GetValueAndDerivativeSingleThreaded(
      parameters, value, derivative );
  }
  else if ( this->m_UseOpenMP )
  {
    return this->GetValueAndDerivativeOpenMP(
      parameters, value, derivative );
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValueAndDerivative itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before
   *   calling GetValueAndDerivative
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValueAndDerivative multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative( parameters );

  /** Initialize some threading related parameters. */
  this->InitializeThreadingParameters();

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative( value, derivative );

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::ThreadedGetValueAndDerivative( ThreadIdType threadId )
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const unsigned int nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType nzji = NonZeroJacobianIndicesType( nnzji );
  DerivativeType imageJacobian = DerivativeType( nzji.size() );
  TransformJacobianType jacobian( FixedImageDimension, nnzji );
  jacobian.Fill( 0.0 ); // needed?

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads
    = static_cast<unsigned long>( vcl_ceil( static_cast<double>( sampleContainerSize )
      / static_cast<double>( this->m_NumberOfThreads ) ) );

  const unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * ( threadId + 1 );
  pos_end = ( pos_end > sampleContainerSize ) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  // also circumvents it?? probably not, since it is simply reference
  // alternative is allocating thread local memory, at the cost of a copy at the end
  DerivativeType & derivative = this->m_ThreaderDerivatives[ threadId ];

  /** Loop over the fixed image to calculate the mean squares. */
  for( threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*threader_fiter).Value().m_ImageCoordinates;
    RealType movingImageValue;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint );

    /** Check if point is inside mask. */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint ); // thread-safe?
    }

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if ( sampleOk )
    {
      sampleOk = this->EvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative );
    }

    if ( sampleOk )
    {
      numberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue
        = static_cast<RealType>( (*threader_fiter).Value().m_ImageValue );

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(
        fixedImageValue, movingImageValue,
        imageJacobian, nzji,
        measure, derivative );
        //this->m_ThreaderDerivatives[ threadID ] );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnessary "false sharing". */
  this->m_ThreaderNumberOfPixelsCounted[ threadId ] = numberOfPixelsCounted;
  this->m_ThreaderValues[ threadId ] = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::AfterThreadedGetValueAndDerivative(
  MeasureType & value, DerivativeType & derivative ) const
{
  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_ThreaderNumberOfPixelsCounted[ 0 ];
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; i++ )
  {
    this->m_NumberOfPixelsCounted += this->m_ThreaderNumberOfPixelsCounted[ i ];
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** The normalization factor. */
  double normal_sum = this->m_NormalizationFactor /
    static_cast<double>( this->m_NumberOfPixelsCounted );

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
  {
    value += this->m_ThreaderValues[ i ];
  }
  value *= normal_sum;

  /** Accumulate derivatives. */
  // it seems that multi-threaded adding is faster than single-threaded
  // it seems that openmp is faster than itk threads
#if 1 // compute single-threadedly
  derivative = this->m_ThreaderDerivatives[ 0 ] * normal_sum;
  for( ThreadIdType i = 1; i < this->m_NumberOfThreads; i++ )
  {
    derivative += this->m_ThreaderDerivatives[ i ] * normal_sum;
  }
#elif 0 // compute multi-threadedly with itk threads
  MultiThreaderComputeDerivativeType * temp = new  MultiThreaderComputeDerivativeType;
  temp->normal_sum = normal_sum;
  temp->m_ThreaderDerivativesIterator = this->m_ThreaderDerivatives.begin();
  temp->derivativeIterator = derivative.begin();
  temp->numberOfParameters = this->GetNumberOfParameters();

  typename ThreaderType::Pointer local_threader = ThreaderType::New();
  local_threader->SetNumberOfThreads( this->m_NumberOfThreadsPerMetric );
  local_threader->SetSingleMethod( ComputeDerivativesThreaderCallback, temp );
  local_threader->SingleMethodExecute();

  delete temp;
#elif 0 // compute multi-threadedly with openmp
  const unsigned int spaceDimension = this->GetNumberOfParameters();
  #pragma omp parallel for
  for( int j = 0; j < spaceDimension; ++j )
  {
    DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
    for( ThreadIdType i = 0; i < this->m_NumberOfThreads; ++i )
    {
      tmp += this->m_ThreaderDerivatives[ i ][ j ];
    }
    derivative[ j ] = tmp * normal_sum;
  }
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 *********** ComputeDerivativesThreaderCallback *************
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::ComputeDerivativesThreaderCallback( void * arg )
{
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );
  ThreadIdType threadID = infoStruct->ThreadID;
  ThreadIdType nrOfThreads = infoStruct->NumberOfThreads;

  MultiThreaderComputeDerivativeType * temp
    = static_cast<MultiThreaderComputeDerivativeType * >( infoStruct->UserData );

  const unsigned int subSize = static_cast<unsigned int>(
    vcl_ceil( static_cast<double>( temp->numberOfParameters )
    / static_cast<double>( nrOfThreads ) ) );
  const unsigned int jmin = threadID * subSize;
  unsigned int jmax = ( threadID + 1 ) * subSize;
  jmax = ( jmax > temp->numberOfParameters ) ? temp->numberOfParameters : jmax;

  for( unsigned int j = jmin; j < jmax; ++j )
  {
    DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
    for( ThreadIdType i = 0; i < nrOfThreads; ++i )
    {
      tmp += temp->m_ThreaderDerivativesIterator[ i ][ j ];
    }
    temp->derivativeIterator[ j ] = tmp * temp->normal_sum;
  }

  return ITK_THREAD_RETURN_VALUE;

} // end ComputeDerivativesThreaderCallback()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::UpdateValueAndDerivativeTerms(
  const RealType fixedImageValue,
  const RealType movingImageValue,
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  MeasureType & measure,
  DerivativeType & deriv ) const
{
  /** The difference squared. */
  const RealType diff = movingImageValue - fixedImageValue;
  const RealType diffdiff = diff * diff;
  measure += diffdiff;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  const RealType diff_2 = diff * 2.0;
  if ( nzji.size() == this->GetNumberOfParameters() )
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator derivit = deriv.begin();
    for ( unsigned int mu = 0; mu < this->GetNumberOfParameters(); ++mu )
    {
      (*derivit) += diff_2 * (*imjacit);
      ++imjacit;
      ++derivit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for ( unsigned int i = 0; i < imageJacobian.GetSize(); ++i )
    {
      const unsigned int index = nzji[ i ];
      deriv[ index ] += diff_2 * imageJacobian[ i ];
    }
  }
} // end UpdateValueAndDerivativeTerms


/**
 * ******************* GetSelfHessian *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetSelfHessian( const TransformParametersType & parameters, HessianType & H ) const
{
  itkDebugMacro("GetSelfHessian()");
  typedef Statistics::MersenneTwisterRandomVariateGenerator RandomGeneratorType;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RandomGeneratorType::Pointer randomGenerator = RandomGeneratorType::GetInstance();
  randomGenerator->Initialize();

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(
    this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices() );
  DerivativeType imageJacobian( nzji.size() );
  TransformJacobianType jacobian;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Prepare Hessian */
  H.set_size( this->GetNumberOfParameters(),
    this->GetNumberOfParameters() );
  //H.Fill(0.0); // done by set_size if sparse matrix

  /** Smooth fixed image */
  typename SmootherType::Pointer smoother = SmootherType::New();
  smoother->SetInput( this->GetFixedImage() );
  smoother->SetSigma( this->GetSelfHessianSmoothingSigma() );
  smoother->Update();

  /** Set up interpolator for fixed image */
  typename FixedImageInterpolatorType::Pointer fixedInterpolator = FixedImageInterpolatorType::New();
  if ( this->m_BSplineInterpolator.IsNotNull() )
  {
    fixedInterpolator->SetSplineOrder( this->m_BSplineInterpolator->GetSplineOrder() );
  }
  else
  {
    fixedInterpolator->SetSplineOrder( 1 );
  }
  fixedInterpolator->SetInputImage( smoother->GetOutput() );

  /** Set up random coordinate sampler
   * Actually we could do without a sampler, but it's easy like this.
   */
  typename SelfHessianSamplerType::Pointer sampler = SelfHessianSamplerType::New();
  //typename DummyFixedImageInterpolatorType::Pointer dummyInterpolator =
  //  DummyFixedImageInterpolatorType::New();
  sampler->SetInputImageRegion( this->GetImageSampler()->GetInputImageRegion() );
  sampler->SetMask( this->GetImageSampler()->GetMask() );
  sampler->SetInput( smoother->GetInput() );
  sampler->SetNumberOfSamples( this->m_NumberOfSamplesForSelfHessian );
  //sampler->SetInterpolator( dummyInterpolator );

  /** Update the imageSampler and get a handle to the sample container. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for ( fiter = fbegin; fiter != fend; ++fiter )
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = (*fiter).Value().m_ImageCoordinates;
    MovingImagePointType mappedPoint;
    MovingImageDerivativeType movingImageDerivative;

    /** Transform point and check if it is inside the B-spline support region. */
    bool sampleOk = this->TransformPoint( fixedPoint, mappedPoint);

    /** Check if point is inside mask. NB: we assume here that the
     * initial transformation is approximately ok.
     */
    if ( sampleOk )
    {
      sampleOk = this->IsInsideMovingMask( mappedPoint );
    }

    /** Check if point is inside moving image. NB: we assume here that the
     * initial transformation is approximately ok.
     */
    if ( sampleOk )
    {
      sampleOk = this->m_Interpolator->IsInsideBuffer( mappedPoint );
    }

    if ( sampleOk )
    {
      this->m_NumberOfPixelsCounted++;

      /** Use the derivative of the fixed image for the self Hessian!
       * \todo: we can do this more efficient without the interpolation,
       * without the sampler, and with a precomputed gradient image,
       * but is this the bottleneck?
       */
      movingImageDerivative = fixedInterpolator->EvaluateDerivative( fixedPoint );
      for ( unsigned int d = 0; d < FixedImageDimension; ++d )
      {
        movingImageDerivative[d] += randomGenerator->GetVariateWithClosedRange(
          this->m_SelfHessianNoiseRange ) - this->m_SelfHessianNoiseRange / 2.0;
      }

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );

      /** Compute this pixel's contribution to the SelfHessian. */
      this->UpdateSelfHessianTerms( imageJacobian, nzji, H );

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(
    sampleContainer->Size(), this->m_NumberOfPixelsCounted );

  /** Compute the measure value and derivative. */
  if ( this->m_NumberOfPixelsCounted > 0 )
  {
    const double normal_sum = 2.0 * this->m_NormalizationFactor /
      static_cast<double>( this->m_NumberOfPixelsCounted );
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
    {
      H.scale_row(i, normal_sum);
    }
  }
  else
  {
    //H.fill_diagonal(1.0);
    for (unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
    {
      H(i,i) = 1.0;
    }
  }

} // end GetSelfHessian()


/**
 * *************** UpdateSelfHessianTerms ***************************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::UpdateSelfHessianTerms(
  const DerivativeType & imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  HessianType & H ) const
{
  typedef typename HessianType::row RowType;
  typedef typename RowType::iterator RowIteratorType;
  typedef typename HessianType::pair_t ElementType;

  // does not work for sparse matrix. \todo: distinguish between sparse and nonsparse
  ///** Do rank-1 update of H */
  //if ( nzji.size() == this->GetNumberOfParameters() )
  //{
  //  /** Loop over all Jacobians. */
  //  vnl_matrix_update( H, imageJacobian, imageJacobian );
  //}
  //else
  //{
    /** Only pick the nonzero Jacobians.
    * Save only upper triangular part of the matrix */
    const unsigned int imjacsize = imageJacobian.GetSize();
    for ( unsigned int i = 0; i < imjacsize; ++i )
    {
      const unsigned int row = nzji[ i ];
      const double imjacrow = imageJacobian[ i ];

      RowType & rowVector = H.get_row( row );
      RowIteratorType rowIt = rowVector.begin();

      for ( unsigned int j = i; j < imjacsize; ++j )
      {
        const unsigned int col = nzji[ j ];
        const double val = imjacrow * imageJacobian[ j ];
        if ( ( val < 1e-14 ) && ( val > -1e-14 ) )
        {
          continue;
        }

        /** The following implements:
         * H(row,col) += imjacrow * imageJacobian[ j ];
         * But more efficient.
         */

        /** Go to next element */
        for (; (rowIt != rowVector.end()) && ((*rowIt).first < col); ++rowIt );

        if ( (rowIt == rowVector.end()) || ((*rowIt).first != col) )
        {
          /** Add new column to the row and set iterator to that column. */
          rowIt = rowVector.insert( rowIt, ElementType( col, val ) );
        }
        else
        {
          /** Add to existing value */
          (*rowIt).second += val;
        }
      }
    }

  //} // end else

} // end UpdateSelfHessianTerms()


} // end namespace itk


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx
