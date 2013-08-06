/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx
#define _itkAdvancedMeanSquaresImageToImageMetric_txx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include "vnl/algo/vnl_matrix_update.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"

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

  this->m_SelfHessianSmoothingSigma = 1.0;
  this->m_NumberOfSamplesForSelfHessian = 100000;

  this->m_SelfHessianNoiseRange = 1.0;

} // end Constructor


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
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template < class TFixedImage, class TMovingImage >
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType & jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType & imageJacobian ) const
{
  typedef typename TransformJacobianType::const_iterator JacobianIteratorType;
  typedef typename DerivativeType::iterator              DerivativeIteratorType;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill( 0.0 );
  const unsigned int sizeImageJacobian = imageJacobian.GetSize();
  for ( unsigned int dim = 0; dim < FixedImageDimension; dim++ )
  {
    const double imDeriv = movingImageDerivative[ dim ];
    DerivativeIteratorType imjac = imageJacobian.begin();

    for ( unsigned int mu = 0; mu < sizeImageJacobian; mu++ )
    {
      (*imjac) += (*jac) * imDeriv;
      ++imjac;
      ++jac;
    }
  }

} // end EvaluateTransformJacobianInnerProduct()


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

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
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
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage,TMovingImage>
::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType & value, DerivativeType & derivative ) const
{
  itkDebugMacro( "GetValueAndDerivative( " << parameters << " ) " );

  typedef typename DerivativeType::ValueType        DerivativeValueType;
  typedef typename TransformJacobianType::ValueType TransformJacobianValueType;

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

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters( parameters );

  /** Update the imageSampler and get a handle to the sample container. */
  this->GetImageSampler()->Update();
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

} // end GetValueAndDerivative()


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
  typedef typename DerivativeType::ValueType        DerivativeValueType;

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

  typedef typename DerivativeType::ValueType        DerivativeValueType;
  typedef typename TransformJacobianType::ValueType TransformJacobianValueType;
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
    for ( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
    {
      H.scale_row(i, normal_sum);
    }
  }
  else
  {
    //H.fill_diagonal(1.0);
    for ( unsigned int i = 0; i < this->GetNumberOfParameters(); ++i )
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


#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_txx

