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
#ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx
#define _itkAdvancedMeanSquaresImageToImageMetric_hxx

#include "itkAdvancedMeanSquaresImageToImageMetric.h"
#include <vnl/algo/vnl_matrix_update.h>
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkComputeImageExtremaFilter.h"

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::AdvancedMeanSquaresImageToImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

  this->m_UseNormalization = false;
  this->m_NormalizationFactor = 1.0;

  /** SelfHessian related variables, experimental feature. */
  this->m_SelfHessianSmoothingSigma = 1.0;
  this->m_SelfHessianNoiseRange = 1.0;
  this->m_NumberOfSamplesForSelfHessian = 100000;

  this->m_SelfHessianNoiseRange = 1.0;

} // end Constructor


/**
 * ********************* Initialize ****************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  if (this->GetUseNormalization())
  {
    /** Try to guess a normalization factor. */
    using ComputeFixedImageExtremaFilterType = typename itk::ComputeImageExtremaFilter<FixedImageType>;
    typename ComputeFixedImageExtremaFilterType::Pointer computeFixedImageExtrema =
      ComputeFixedImageExtremaFilterType::New();
    computeFixedImageExtrema->SetInput(this->GetFixedImage());
    computeFixedImageExtrema->SetImageRegion(this->GetFixedImageRegion());
    if (this->m_FixedImageMask.IsNotNull())
    {
      computeFixedImageExtrema->SetUseMask(true);
      const FixedImageMaskSpatialObject2Type * fmask =
        dynamic_cast<const FixedImageMaskSpatialObject2Type *>(this->m_FixedImageMask.GetPointer());
      if (fmask)
      {
        computeFixedImageExtrema->SetImageSpatialMask(fmask);
      }
      else
      {
        computeFixedImageExtrema->SetImageMask(this->GetFixedImageMask());
      }
    }

    computeFixedImageExtrema->Update();

    this->m_FixedImageTrueMax = computeFixedImageExtrema->GetMaximum();
    this->m_FixedImageTrueMin = computeFixedImageExtrema->GetMinimum();

    this->m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
      this->m_FixedImageTrueMin -
      this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));
    this->m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
      this->m_FixedImageTrueMax +
      this->m_FixedLimitRangeRatio * (this->m_FixedImageTrueMax - this->m_FixedImageTrueMin));

    using ComputeMovingImageExtremaFilterType = typename itk::ComputeImageExtremaFilter<MovingImageType>;
    typename ComputeMovingImageExtremaFilterType::Pointer computeMovingImageExtrema =
      ComputeMovingImageExtremaFilterType::New();
    computeMovingImageExtrema->SetInput(this->GetMovingImage());
    computeMovingImageExtrema->SetImageRegion(this->GetMovingImage()->GetBufferedRegion());
    if (this->m_MovingImageMask.IsNotNull())
    {
      computeMovingImageExtrema->SetUseMask(true);
      const MovingImageMaskSpatialObject2Type * mMask =
        dynamic_cast<const MovingImageMaskSpatialObject2Type *>(this->m_MovingImageMask.GetPointer());
      if (mMask)
      {
        computeMovingImageExtrema->SetImageSpatialMask(mMask);
      }
      else
      {
        computeMovingImageExtrema->SetImageMask(this->GetMovingImageMask());
      }
    }

    computeMovingImageExtrema->Update();

    this->m_MovingImageTrueMax = computeMovingImageExtrema->GetMaximum();
    this->m_MovingImageTrueMin = computeMovingImageExtrema->GetMinimum();

    this->m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
      this->m_MovingImageTrueMin -
      this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));
    this->m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
      this->m_MovingImageTrueMax +
      this->m_MovingLimitRangeRatio * (this->m_MovingImageTrueMax - this->m_MovingImageTrueMin));

    // TODO: we may actually reuse these values from AdvancedImageToImageMetric::InitializeLimiters
    // without recomputing them here.
    const double diff1 = this->m_FixedImageTrueMax - this->m_MovingImageTrueMin;
    const double diff2 = this->m_MovingImageTrueMax - this->m_FixedImageTrueMin;
    const double maxdiff = std::max(diff1, diff2);

    /** We guess that maxdiff/10 is the maximum average difference that will
     * be observed.
     * \todo We may involve the standard derivation of the image into this estimate.
     */
    this->m_NormalizationFactor = 1.0;
    if (maxdiff > 1e-10)
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

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << "UseNormalization: " << this->m_UseNormalization << std::endl;
  os << "SelfHessianSmoothingSigma: " << this->m_SelfHessianSmoothingSigma << std::endl;
  os << "NumberOfSamplesForSelfHessian: " << this->m_NumberOfSamplesForSelfHessian << std::endl;

} // end PrintSelf()


/**
 * ******************* GetValueSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
auto
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetValueSingleThreaded(
  const TransformParametersType & parameters) const -> MeasureType
{
  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

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
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image samples to calculate the mean squares. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
    }

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<double>(fiter->Value().m_ImageValue);

      /** The difference squared. */
      const RealType diff = movingImageValue - fixedImageValue;
      measure += diff * diff;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Update measure value. */
  double normal_sum = 0.0;
  if (this->m_NumberOfPixelsCounted > 0)
  {
    normal_sum = this->m_NormalizationFactor / static_cast<double>(this->m_NumberOfPixelsCounted);
  }
  measure *= normal_sum;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValueSingleThreaded()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueSingleThreaded(parameters);
  }

  /** Call non-thread-safe stuff, such as:
   *   this->SetTransformParameters( parameters );
   *   this->GetImageSampler()->Update();
   * Because of these calls GetValue itself is not thread-safe,
   * so cannot be called multiple times simultaneously.
   * This is however needed in the CombinationImageToImageMetric.
   * In that case, you need to:
   * - switch the use of this function to on, using m_UseMetricSingleThreaded = true
   * - call BeforeThreadedGetValueAndDerivative once (single-threaded) before calling GetValue
   * - switch the use of this function to off, using m_UseMetricSingleThreaded = false
   * - Now you can call GetValue multi-threaded.
   */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Launch multi-threading metric */
  this->LaunchGetValueThreaderCallback();

  /** Gather the metric values from all threads. */
  MeasureType value = NumericTraits<MeasureType>::Zero;
  this->AfterThreadedGetValue(value);

  return value;

} // end GetValue()


/**
 * ******************* ThreadedGetValue *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValue(ThreadIdType threadId)
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType   measure = NumericTraits<MeasureType>::Zero;

  /** Loop over the fixed image to calculate the mean squares. */
  for (threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = threader_fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value M(T(x)) and check if
     * the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->FastEvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr, threadId);
    }

    if (sampleOk)
    {
      ++numberOfPixelsCounted;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(threader_fiter->Value().m_ImageValue);

      /** The difference squared. */
      const RealType diff = movingImageValue - fixedImageValue;
      measure += diff * diff;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;

} // end ThreadedGetValue()


/**
 * ******************* AfterThreadedGetValue *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValue(MeasureType & value) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_GetValueAndDerivativePerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    this->m_NumberOfPixelsCounted += this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** The normalization factor. */
  DerivativeValueType normal_sum =
    this->m_NormalizationFactor / static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }
  value *= normal_sum;

} // end AfterThreadedGetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
  const TransformParametersType & parameters,
  DerivativeType &                derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

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
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk =
        this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative);
    }

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

#if 0
      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );
#else
      /** Compute the inner product of the transform Jacobian and the moving image gradient. */
      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct(
        fixedPoint, movingImageDerivative, imageJacobian, nzji);
#endif

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(fixedImageValue, movingImageValue, imageJacobian, nzji, measure, derivative);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the measure value and derivative. */
  double normal_sum = 0.0;
  if (this->m_NumberOfPixelsCounted > 0)
  {
    normal_sum = this->m_NormalizationFactor / static_cast<double>(this->m_NumberOfPixelsCounted);
  }
  measure *= normal_sum;
  derivative *= normal_sum;

  /** The return value. */
  value = measure;

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
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
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative(value, derivative);

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(ThreadIdType threadId)
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji = NonZeroJacobianIndicesType(nnzji);
  DerivativeType               imageJacobian(nnzji);

  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Derivative;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  unsigned long pos_begin = nrOfSamplesPerThreads * threadId;
  unsigned long pos_end = nrOfSamplesPerThreads * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

  /** Create variables to store intermediate results. circumvent false sharing */
  unsigned long numberOfPixelsCounted = 0;
  MeasureType   measure = NumericTraits<MeasureType>::Zero;

  /** Loop over the fixed image to calculate the mean squares. */
  for (threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = threader_fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->FastEvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative, threadId);
    }

    if (sampleOk)
    {
      ++numberOfPixelsCounted;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(threader_fiter->Value().m_ImageValue);

#if 0
      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );
#else
      /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
      this->m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct(
        fixedPoint, movingImageDerivative, imageJacobian, nzji);
#endif

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(fixedImageValue, movingImageValue, imageJacobian, nzji, measure, derivative);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_GetValueAndDerivativePerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    this->m_NumberOfPixelsCounted += this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** The normalization factor. */
  DerivativeValueType normal_sum =
    this->m_NormalizationFactor / static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }
  value *= normal_sum;

  /** Accumulate derivatives. */
  // compute single-threadedly
  if (!this->m_UseMultiThread && false) // force multi-threaded
  {
    derivative = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative * normal_sum;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      derivative += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative * normal_sum;
    }
  }
  // compute multi-threadedly with itk threads
  else if (true) // force ITK threads !this->m_UseOpenMP )
  {
    this->m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
    this->m_ThreaderMetricParameters.st_NormalizationFactor = 1.0 / normal_sum;

    this->m_Threader->SetSingleMethod(this->AccumulateDerivativesThreaderCallback,
                                      const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));
    this->m_Threader->SingleMethodExecute();
  }
#ifdef ELASTIX_USE_OPENMP
  // compute multi-threadedly with openmp
  else
  {
    const int spaceDimension = static_cast<int>(this->GetNumberOfParameters());

#  pragma omp parallel for
    for (int j = 0; j < spaceDimension; ++j)
    {
      DerivativeValueType tmp = NumericTraits<DerivativeValueType>::Zero;
      for (ThreadIdType i = 0; i < numberOfThreads; ++i)
      {
        tmp += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j];
      }
      derivative[j] = tmp * normal_sum;
    }
  }
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::UpdateValueAndDerivativeTerms(
  const RealType                     fixedImageValue,
  const RealType                     movingImageValue,
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  MeasureType &                      measure,
  DerivativeType &                   deriv) const
{
  /** The difference squared. */
  const RealType diff = movingImageValue - fixedImageValue;
  const RealType diffdiff = diff * diff;
  measure += diffdiff;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  const RealType diff_2 = diff * 2.0;

  const auto numberOfParameters = this->GetNumberOfParameters();

  if (nzji.size() == numberOfParameters)
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator       derivit = deriv.begin();
    for (unsigned int mu = 0; mu < numberOfParameters; ++mu)
    {
      (*derivit) += diff_2 * (*imjacit);
      ++imjacit;
      ++derivit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for (unsigned int i = 0; i < imageJacobian.GetSize(); ++i)
    {
      const unsigned int index = nzji[i];
      deriv[index] += diff_2 * imageJacobian[i];
    }
  }
} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetSelfHessian *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::GetSelfHessian(
  const TransformParametersType & parameters,
  HessianType &                   H) const
{
  itkDebugMacro("GetSelfHessian()");
  using RandomGeneratorType = Statistics::MersenneTwisterRandomVariateGenerator;

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  RandomGeneratorType::Pointer randomGenerator = RandomGeneratorType::GetInstance();
  randomGenerator->Initialize();

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Prepare Hessian */
  H.set_size(numberOfParameters, numberOfParameters);
  // H.Fill(0.0); // done by set_size if sparse matrix

  /** Smooth fixed image */
  auto smoother = SmootherType::New();
  smoother->SetInput(this->GetFixedImage());
  smoother->SetSigma(this->GetSelfHessianSmoothingSigma());
  smoother->Update();

  /** Set up interpolator for fixed image */
  auto fixedInterpolator = FixedImageInterpolatorType::New();
  if (this->m_BSplineInterpolator.IsNotNull())
  {
    fixedInterpolator->SetSplineOrder(this->m_BSplineInterpolator->GetSplineOrder());
  }
  else
  {
    fixedInterpolator->SetSplineOrder(1);
  }
  fixedInterpolator->SetInputImage(smoother->GetOutput());

  /** Set up random coordinate sampler
   * Actually we could do without a sampler, but it's easy like this.
   */
  auto sampler = SelfHessianSamplerType::New();
  // typename DummyFixedImageInterpolatorType::Pointer dummyInterpolator =
  //  DummyFixedImageInterpolatorType::New();
  sampler->SetInputImageRegion(this->GetImageSampler()->GetInputImageRegion());
  sampler->SetMask(this->GetImageSampler()->GetMask());
  sampler->SetInput(smoother->GetInput());
  sampler->SetNumberOfSamples(this->m_NumberOfSamplesForSelfHessian);
  // sampler->SetInterpolator( dummyInterpolator );

  /** Update the imageSampler and get a handle to the sample container. */
  sampler->Update();
  ImageSampleContainerPointer sampleContainer = sampler->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the mean squares. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Check if point is inside moving image. NB: we assume here that the
     * initial transformation is approximately ok.
     */
    if (sampleOk)
    {
      sampleOk = this->m_Interpolator->IsInsideBuffer(mappedPoint);
    }

    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Use the derivative of the fixed image for the self Hessian!
       * \todo: we can do this more efficient without the interpolation,
       * without the sampler, and with a precomputed gradient image,
       * but is this the bottleneck?
       */
      movingImageDerivative = fixedInterpolator->EvaluateDerivative(fixedPoint);
      for (unsigned int d = 0; d < FixedImageDimension; ++d)
      {
        movingImageDerivative[d] += randomGenerator->GetVariateWithClosedRange(this->m_SelfHessianNoiseRange) -
                                    this->m_SelfHessianNoiseRange / 2.0;
      }

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Compute this pixel's contribution to the SelfHessian. */
      this->UpdateSelfHessianTerms(imageJacobian, nzji, H);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the measure value and derivative. */
  if (this->m_NumberOfPixelsCounted > 0)
  {
    const double normal_sum = 2.0 * this->m_NormalizationFactor / static_cast<double>(this->m_NumberOfPixelsCounted);
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      H.scale_row(i, normal_sum);
    }
  }
  else
  {
    // H.fill_diagonal(1.0);
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      H(i, i) = 1.0;
    }
  }

} // end GetSelfHessian()


/**
 * *************** UpdateSelfHessianTerms ***************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedMeanSquaresImageToImageMetric<TFixedImage, TMovingImage>::UpdateSelfHessianTerms(
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  HessianType &                      H) const
{
  using RowType = typename HessianType::row;
  using RowIteratorType = typename RowType::iterator;
  using ElementType = typename HessianType::pair_t;

  // does not work for sparse matrix. \todo: distinguish between sparse and nonsparse
  ///** Do rank-1 update of H */
  // if ( nzji.size() == this->GetNumberOfParameters() )
  //{
  //  /** Loop over all Jacobians. */
  //  vnl_matrix_update( H, imageJacobian, imageJacobian );
  //}
  // else
  //{
  /** Only pick the nonzero Jacobians.
   * Save only upper triangular part of the matrix */
  const unsigned int imjacsize = imageJacobian.GetSize();
  for (unsigned int i = 0; i < imjacsize; ++i)
  {
    const unsigned int row = nzji[i];
    const double       imjacrow = imageJacobian[i];

    RowType &       rowVector = H.get_row(row);
    RowIteratorType rowIt = rowVector.begin();

    for (unsigned int j = i; j < imjacsize; ++j)
    {
      const unsigned int col = nzji[j];
      const double       val = imjacrow * imageJacobian[j];
      if ((val < 1e-14) && (val > -1e-14))
      {
        continue;
      }

      /** The following implements:
       * H(row,col) += imjacrow * imageJacobian[ j ];
       * But more efficient.
       */

      /** Go to next element */
      for (; (rowIt != rowVector.end()) && (rowIt->first < col); ++rowIt)
      {
      }

      if ((rowIt == rowVector.end()) || (rowIt->first != col))
      {
        /** Add new column to the row and set iterator to that column. */
        rowIt = rowVector.insert(rowIt, ElementType(col, val));
      }
      else
      {
        /** Add to existing value */
        rowIt->second += val;
      }
    }
  }

  //} // end else

} // end UpdateSelfHessianTerms()


} // end namespace itk

#endif // end #ifndef _itkAdvancedMeanSquaresImageToImageMetric_hxx
