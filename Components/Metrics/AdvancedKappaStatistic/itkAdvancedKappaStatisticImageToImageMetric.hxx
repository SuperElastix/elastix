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
#ifndef _itkAdvancedKappaStatisticImageToImageMetric_hxx
#define _itkAdvancedKappaStatisticImageToImageMetric_hxx

#include "itkAdvancedKappaStatisticImageToImageMetric.h"
#include <cmath> // For abs.

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::AdvancedKappaStatisticImageToImageMetric()
{
  this->SetComputeGradient(true);
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

  this->m_UseForegroundValue = true; // for backwards compatibility
  this->m_ForegroundValue = 1.0;
  this->m_Epsilon = 1e-3;
  this->m_Complement = true;

} // end Constructor


/**
 * ******************* InitializeThreadingParameters *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Only resize the array of structs when needed. */
  m_KappaGetValueAndDerivativePerThreadVariables.resize(numberOfThreads);

  /** Some initialization. */
  const SizeValueType       zero1 = NumericTraits<SizeValueType>::Zero;
  const DerivativeValueType zero2 = NumericTraits<DerivativeValueType>::Zero;
  for (auto & perThreadVariable : m_KappaGetValueAndDerivativePerThreadVariables)
  {
    perThreadVariable.st_NumberOfPixelsCounted = zero1;
    perThreadVariable.st_AreaSum = zero1;
    perThreadVariable.st_AreaIntersection = zero1;
    perThreadVariable.st_DerivativeSum1.SetSize(this->GetNumberOfParameters());
    perThreadVariable.st_DerivativeSum2.SetSize(this->GetNumberOfParameters());
    perThreadVariable.st_DerivativeSum1.Fill(zero2);
    perThreadVariable.st_DerivativeSum2.Fill(zero2);
  }

} // end InitializeThreadingParameters()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "UseForegroundValue: " << (this->m_UseForegroundValue ? "On" : "Off") << std::endl;
  os << indent << "Complement: " << (this->m_Complement ? "On" : "Off") << std::endl;
  os << indent << "ForegroundValue: " << this->m_ForegroundValue << std::endl;
  os << indent << "Epsilon: " << this->m_Epsilon << std::endl;

} // end PrintSelf()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

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

  /** Some variables. */
  RealType    movingImageValue;
  std::size_t fixedForegroundArea = 0; // or unsigned long
  std::size_t movingForegroundArea = 0;
  std::size_t intersection = 0;

  /** Loop over the fixed image samples to calculate the kappa statistic. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if point is inside moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
    }

    /** Do the actual calculation of the metric value. */
    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

      /** Update the intermediate values. */
      if (this->m_UseForegroundValue)
      {
        const RealType diffFixed = std::abs(fixedImageValue - this->m_ForegroundValue);
        const RealType diffMoving = std::abs(movingImageValue - this->m_ForegroundValue);
        if (diffFixed < this->m_Epsilon)
        {
          ++fixedForegroundArea;
        }
        if (diffMoving < this->m_Epsilon)
        {
          ++movingForegroundArea;
        }
        if (diffFixed < this->m_Epsilon && diffMoving < this->m_Epsilon)
        {
          ++intersection;
        }
      }
      else
      {
        if (fixedImageValue > this->m_Epsilon)
        {
          ++fixedForegroundArea;
        }
        if (movingImageValue > this->m_Epsilon)
        {
          ++movingForegroundArea;
        }
        if (fixedImageValue > this->m_Epsilon && movingImageValue > this->m_Epsilon)
        {
          ++intersection;
        }
      }

    } // end if samplOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the final metric value. */
  std::size_t areaSum = fixedForegroundArea + movingForegroundArea;
  if (areaSum == 0)
  {
    measure = NumericTraits<MeasureType>::Zero;
  }
  else
  {
    measure = 1.0 - 2.0 * static_cast<MeasureType>(intersection) / static_cast<MeasureType>(areaSum);
  }
  if (!this->m_Complement)
  {
    measure = 1.0 - measure;
  }

  /** Return the mean squares measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
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
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());

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

  /** Some variables. */
  RealType    movingImageValue;
  std::size_t fixedForegroundArea = 0; // or unsigned long
  std::size_t movingForegroundArea = 0;
  std::size_t intersection = 0;

  DerivativeType vecSum1(this->GetNumberOfParameters());
  DerivativeType vecSum2(this->GetNumberOfParameters());
  vecSum1.Fill(NumericTraits<DerivativeValueType>::ZeroValue());
  vecSum2.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over the fixed image to calculate the kappa statistic. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    MovingImageDerivativeType movingImageDerivative;
    if (sampleOk)
    {
      sampleOk =
        this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative);
    }

    /** Do the actual calculation of the metric value. */
    if (sampleOk)
    {
      this->m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(fixedImageValue,
                                          movingImageValue,
                                          fixedForegroundArea,
                                          movingForegroundArea,
                                          intersection,
                                          imageJacobian,
                                          nzji,
                                          vecSum1,
                                          vecSum2);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the final metric value. */
  std::size_t       areaSum = fixedForegroundArea + movingForegroundArea;
  const MeasureType intersectionFloat = static_cast<MeasureType>(intersection);
  const MeasureType areaSumFloat = static_cast<MeasureType>(areaSum);
  if (areaSum > 0)
  {
    measure = 1.0 - 2.0 * intersectionFloat / areaSumFloat;
  }
  if (!this->m_Complement)
  {
    measure = 1.0 - measure;
  }
  value = measure;

  /** Calculate the derivative. */
  MeasureType direction = -1.0;
  if (!this->m_Complement)
  {
    direction = 1.0;
  }
  const MeasureType areaSumFloatSquare = direction * areaSumFloat * areaSumFloat;
  const MeasureType tmp1 = areaSumFloat / areaSumFloatSquare;
  const MeasureType tmp2 = 2.0 * intersectionFloat / areaSumFloatSquare;

  if (areaSum > 0)
  {
    derivative = tmp1 * vecSum1 - tmp2 * vecSum2;
  }
  else
  {
    derivative.Fill(0.0);
  }

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
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
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(
  ThreadIdType threadId)
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji = NonZeroJacobianIndicesType(nnzji);
  DerivativeType               imageJacobian(nzji.size());

  /** Get handles to the pre-allocated derivatives for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & vecSum1 = this->m_KappaGetValueAndDerivativePerThreadVariables[threadId].st_DerivativeSum1;
  DerivativeType & vecSum2 = this->m_KappaGetValueAndDerivativePerThreadVariables[threadId].st_DerivativeSum2;

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

  /** Some variables. */
  RealType      movingImageValue;
  std::size_t   fixedForegroundArea = 0; // or unsigned long
  std::size_t   movingForegroundArea = 0;
  std::size_t   intersection = 0;
  unsigned long numberOfPixelsCounted = 0;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->Begin();
  fbegin += (int)pos_begin;
  fend += (int)pos_end;

  /** Loop over the fixed image to calculate the kappa statistic. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value M(T(x)) and derivative dM/dx and check if
     * the point is inside the moving image buffer.
     */
    MovingImageDerivativeType movingImageDerivative;
    if (sampleOk)
    {
      sampleOk = this->FastEvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative, threadId);
    }

    /** Do the actual calculation of the metric value. */
    if (sampleOk)
    {
      ++numberOfPixelsCounted;

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

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
      this->UpdateValueAndDerivativeTerms(fixedImageValue,
                                          movingImageValue,
                                          fixedForegroundArea,
                                          movingForegroundArea,
                                          intersection,
                                          imageJacobian,
                                          nzji,
                                          vecSum1,
                                          vecSum2);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_KappaGetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_KappaGetValueAndDerivativePerThreadVariables[threadId].st_AreaSum =
    fixedForegroundArea + movingForegroundArea;
  this->m_KappaGetValueAndDerivativePerThreadVariables[threadId].st_AreaIntersection = intersection;

} // end GetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  this->m_NumberOfPixelsCounted = this->m_KappaGetValueAndDerivativePerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    this->m_NumberOfPixelsCounted += this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;

    /** Reset this variable for the next iteration. */
    this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = 0;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Accumulate values. */
  const MeasureType zero = NumericTraits<MeasureType>::Zero;
  MeasureType       areaSum = zero;
  MeasureType       intersection = zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    areaSum += this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_AreaSum;
    intersection += this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_AreaIntersection;

    /** Reset these variables for the next iteration. */
    this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_AreaSum = zero;
    this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_AreaIntersection = zero;
  }

  if (areaSum == 0)
  {
    return;
  }

  /** Compute the final metric value. */
  value = 1.0 - 2.0 * intersection / areaSum;
  if (!this->m_Complement)
  {
    value = 1.0 - value;
  }

  /** Some intermediate values to calculate the derivative. */
  MeasureType direction = -1.0;
  if (!this->m_Complement)
  {
    direction = 1.0;
  }
  const MeasureType areaSumSquare = direction * areaSum * areaSum;
  const MeasureType tmp1 = direction / areaSum;
  const MeasureType tmp2 = 2.0 * intersection / areaSumSquare;

  /** Accumulate intermediate values and calculate derivative. */
  if (!this->m_UseMultiThread) // single-threaded
  {
    DerivativeType vecSum1 = this->m_KappaGetValueAndDerivativePerThreadVariables[0].st_DerivativeSum1;
    DerivativeType vecSum2 = this->m_KappaGetValueAndDerivativePerThreadVariables[0].st_DerivativeSum2;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      vecSum1 += this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_DerivativeSum1;
      vecSum2 += this->m_KappaGetValueAndDerivativePerThreadVariables[i].st_DerivativeSum2;
    }
    derivative = tmp1 * vecSum1 - tmp2 * vecSum2;
  }
  else // multi-threaded
  {
    MultiThreaderAccumulateDerivativeType * temp = new MultiThreaderAccumulateDerivativeType;

    temp->st_Metric = const_cast<Self *>(this);
    temp->st_Coefficient1 = tmp1;
    temp->st_Coefficient2 = tmp2;
    temp->st_DerivativePointer = derivative.begin();

    this->m_Threader->SetSingleMethod(AccumulateDerivativesThreaderCallback, temp);
    this->m_Threader->SingleMethodExecute();

    delete temp;
  }

} // end AfterThreadedGetValueAndDerivative()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::AccumulateDerivativesThreaderCallback(void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadId = infoStruct->WorkUnitID;
  ThreadIdType     nrOfThreads = infoStruct->NumberOfWorkUnits;

  MultiThreaderAccumulateDerivativeType * temp =
    static_cast<MultiThreaderAccumulateDerivativeType *>(infoStruct->UserData);

  const unsigned int numPar = temp->st_Metric->GetNumberOfParameters();
  const unsigned int subSize =
    static_cast<unsigned int>(std::ceil(static_cast<double>(numPar) / static_cast<double>(nrOfThreads)));
  unsigned int jmin = threadId * subSize;
  unsigned int jmax = (threadId + 1) * subSize;
  jmax = (jmax > numPar) ? numPar : jmax;

  const DerivativeValueType zero = NumericTraits<DerivativeValueType>::Zero;
  DerivativeValueType       sum1, sum2;
  for (unsigned int j = jmin; j < jmax; ++j)
  {
    sum1 = sum2 = zero;
    for (auto & perThreadVariable : temp->st_Metric->m_KappaGetValueAndDerivativePerThreadVariables)
    {
      sum1 += perThreadVariable.st_DerivativeSum1[j];
      sum2 += perThreadVariable.st_DerivativeSum2[j];

      /** Reset these variables for the next iteration. */
      perThreadVariable.st_DerivativeSum1[j] = zero;
      perThreadVariable.st_DerivativeSum2[j] = zero;
    }
    temp->st_DerivativePointer[j] = temp->st_Coefficient1 * sum1 - temp->st_Coefficient2 * sum2;
  }

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AccumulateDerivativesThreaderCallback()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::UpdateValueAndDerivativeTerms(
  const RealType &                   fixedImageValue,
  const RealType &                   movingImageValue,
  std::size_t &                      fixedForegroundArea,
  std::size_t &                      movingForegroundArea,
  std::size_t &                      intersection,
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType &                   sum1,
  DerivativeType &                   sum2) const
{
  /** Update the intermediate values. */
  bool usableFixedSample = false;
  if (this->m_UseForegroundValue)
  {
    const RealType diffFixed = std::abs(fixedImageValue - this->m_ForegroundValue);
    const RealType diffMoving = std::abs(movingImageValue - this->m_ForegroundValue);
    if (diffFixed < this->m_Epsilon)
    {
      ++fixedForegroundArea;
      usableFixedSample = true;
    }
    if (diffMoving < this->m_Epsilon)
    {
      ++movingForegroundArea;
    }
    if (diffFixed < this->m_Epsilon && diffMoving < this->m_Epsilon)
    {
      ++intersection;
    }
  }
  else
  {
    if (fixedImageValue > this->m_Epsilon)
    {
      ++fixedForegroundArea;
      usableFixedSample = true;
    }
    if (movingImageValue > this->m_Epsilon)
    {
      ++movingForegroundArea;
    }
    if (fixedImageValue > this->m_Epsilon && movingImageValue > this->m_Epsilon)
    {
      ++intersection;
    }
  }

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if (nzji.size() == numberOfParameters)
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator       sum1it = sum1.begin();
    typename DerivativeType::iterator       sum2it = sum2.begin();
    for (unsigned int mu = 0; mu < numberOfParameters; ++mu)
    {
      if (usableFixedSample)
      {
        (*sum1it) += 2.0 * (*imjacit);
      }
      (*sum2it) += (*imjacit);

      /** Increase iterators. */
      ++imjacit;
      ++sum1it;
      ++sum2it;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for (unsigned int i = 0; i < nzji.size(); ++i)
    {
      const unsigned int        index = nzji[i];
      const DerivativeValueType imjac = imageJacobian[i];
      if (usableFixedSample)
      {
        sum1[index] += 2.0 * imjac;
      }
      sum2[index] += imjac;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * *************** ComputeGradient ***************************
 */

template <class TFixedImage, class TMovingImage>
void
AdvancedKappaStatisticImageToImageMetric<TFixedImage, TMovingImage>::ComputeGradient()
{
  /** Typedefs. */
  using GradientIteratorType = itk::ImageRegionIteratorWithIndex<GradientImageType>;
  using MovingIteratorType = itk::ImageRegionConstIteratorWithIndex<MovingImageType>;

  /** Create a temporary moving gradient image. */
  auto tempGradientImage = GradientImageType::New();
  tempGradientImage->SetRegions(this->m_MovingImage->GetBufferedRegion().GetSize());
  tempGradientImage->Allocate();

  /** Create and reset iterators. */
  GradientIteratorType git(tempGradientImage, tempGradientImage->GetBufferedRegion());
  MovingIteratorType   mit(this->m_MovingImage, this->m_MovingImage->GetBufferedRegion());
  git.GoToBegin();
  mit.GoToBegin();

  /** Some temporary variables. */
  typename MovingImageType::IndexType   minusIndex, plusIndex, currIndex;
  typename GradientImageType::PixelType tempGradPixel;
  typename MovingImageType::SizeType    movingSize = this->m_MovingImage->GetBufferedRegion().GetSize();
  typename MovingImageType::IndexType   movingIndex = this->m_MovingImage->GetBufferedRegion().GetIndex();

  /** Loop over the images. */
  while (!mit.IsAtEnd())
  {
    /** Get the current index. */
    currIndex = mit.GetIndex();
    minusIndex = currIndex;
    plusIndex = currIndex;
    for (unsigned int i = 0; i < MovingImageDimension; ++i)
    {
      /** Check for being on the edge of the moving image. */
      if (currIndex[i] == movingIndex[i] || currIndex[i] == static_cast<int>(movingIndex[i] + movingSize[i] - 1))
      {
        tempGradPixel[i] = 0.0;
      }
      else
      {
        /** Get the left, center and right values. */
        minusIndex[i] = currIndex[i] - 1;
        plusIndex[i] = currIndex[i] + 1;
        const RealType minusVal = static_cast<RealType>(this->m_MovingImage->GetPixel(minusIndex));
        const RealType plusVal = static_cast<RealType>(this->m_MovingImage->GetPixel(plusIndex));
        const RealType minusDiff = std::abs(minusVal - this->m_ForegroundValue);
        const RealType plusDiff = std::abs(plusVal - this->m_ForegroundValue);

        /** Calculate the gradient. */
        if (minusDiff >= this->m_Epsilon && plusDiff < this->m_Epsilon)
        {
          tempGradPixel[i] = 1.0;
        }
        else if (minusDiff < this->m_Epsilon && plusDiff >= this->m_Epsilon)
        {
          tempGradPixel[i] = -1.0;
        }
        else
        {
          tempGradPixel[i] = 0.0;
        }
      }

      /** Reset indices. */
      minusIndex = currIndex;
      plusIndex = currIndex;

    } // end for loop

    /** Set the gradient value and increase iterators. */
    git.Set(tempGradPixel);
    ++git;
    ++mit;

  } // end while loop

  this->m_GradientImage = tempGradientImage;

} // end ComputeGradient()


} // end namespace itk

#endif // end #ifndef _itkAdvancedKappaStatisticImageToImageMetric_txx
