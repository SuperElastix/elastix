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
#ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx
#define _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx

#include "itkAdvancedNormalizedCorrelationImageToImageMetric.h"

#include <algorithm> // For min.
#include <cassert>

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <typename TFixedImage, typename TMovingImage>
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage,
                                                TMovingImage>::AdvancedNormalizedCorrelationImageToImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);

} // end Constructor


/**
 * ******************* InitializeThreadingParameters *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   * Filling the potentially large vectors is performed later, in each thread,
   * which has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  m_CorrelationGetValueAndDerivativePerThreadVariables.resize(numberOfThreads);

  /** Some initialization. */
  const auto numberOfParameters = this->GetNumberOfParameters();
  for (auto & perThreadVariable : m_CorrelationGetValueAndDerivativePerThreadVariables)
  {
    perThreadVariable.st_NumberOfPixelsCounted = SizeValueType{};
    perThreadVariable.st_Sff = 0.0;
    perThreadVariable.st_Smm = 0.0;
    perThreadVariable.st_Sfm = 0.0;
    perThreadVariable.st_Sf = 0.0;
    perThreadVariable.st_Sm = 0.0;
    perThreadVariable.st_DerivativeF.SetSize(numberOfParameters);
    perThreadVariable.st_DerivativeM.SetSize(numberOfParameters);
    perThreadVariable.st_Differential.SetSize(this->GetNumberOfParameters());
    perThreadVariable.st_DerivativeF.Fill(0.0);
    perThreadVariable.st_DerivativeM.Fill(0.0);
    perThreadVariable.st_Differential.Fill(0.0);
  }

} // end InitializeThreadingParameters()


/**
 * *************** UpdateDerivativeTerms ***************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::UpdateDerivativeTerms(
  const RealType                     fixedImageValue,
  const RealType                     movingImageValue,
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType &                   derivativeF,
  DerivativeType &                   derivativeM,
  DerivativeType &                   differential) const
{
  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  if (nzji.size() == numberOfParameters)
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::iterator       derivativeFit = derivativeF.begin();
    typename DerivativeType::iterator       derivativeMit = derivativeM.begin();
    typename DerivativeType::iterator       differentialit = differential.begin();

    for (unsigned int mu = 0; mu < numberOfParameters; ++mu)
    {
      (*derivativeFit) += fixedImageValue * (*imjacit);
      (*derivativeMit) += movingImageValue * (*imjacit);
      (*differentialit) += (*imjacit);
      ++imjacit;
      ++derivativeFit;
      ++derivativeMit;
      ++differentialit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for (unsigned int i = 0; i < imageJacobian.GetSize(); ++i)
    {
      const unsigned int index = nzji[i];
      const RealType     differentialtmp = imageJacobian[i];
      derivativeF[index] += fixedImageValue * differentialtmp;
      derivativeM[index] += movingImageValue * differentialtmp;
      differential[index] += differentialtmp;
    }
  }

} // end UpdateValueAndDerivativeTerms()


/**
 * ******************* GetValue *******************
 */

template <typename TFixedImage, typename TMovingImage>
auto
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const ParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Initialize some variables */
  Superclass::m_NumberOfPixelsCounted = 0;
  MeasureType measure{};

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

  /** Create variables to store intermediate results. */
  AccumulateType sff{};
  AccumulateType smm{};
  AccumulateType sfm{};
  AccumulateType sf{};
  AccumulateType sm{};

  /** Loop over the fixed image samples to calculate the mean squares. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fixedImageSample.m_ImageCoordinates;
    RealType                    movingImageValue;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value and check if the point is
     * inside the moving image buffer. */
    if (sampleOk)
    {
      sampleOk = this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, nullptr);
    }

    if (sampleOk)
    {
      Superclass::m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const auto fixedImageValue = static_cast<double>(fixedImageSample.m_ImageValue);

      /** Update some sums needed to calculate NC. */
      sff += fixedImageValue * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue * movingImageValue;
      sf += fixedImageValue;
      sm += movingImageValue;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** If NumberOfPixelsCounted > 0, then subtract things from sff, smm and sfm. */
  const auto N = static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);
  if (Superclass::m_NumberOfPixelsCounted > 0)
  {
    sff -= (sf * sf / N);
    smm -= (sm * sm / N);
    sfm -= (sf * sm / N);
  }

  /** The denominator of the NC. */
  const RealType denom = -1.0 * std::sqrt(sff * smm);

  /** Calculate the measure value. */
  if (Superclass::m_NumberOfPixelsCounted > 0 && denom < -1e-14)
  {
    measure = sfm / denom;
  }
  else
  {
    measure = MeasureType{};
  }

  /** Return the NC measure value. */
  return measure;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
  const ParametersType & parameters,
  DerivativeType &       derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables. */
  Superclass::m_NumberOfPixelsCounted = 0;
  derivative.set_size(this->GetNumberOfParameters());
  derivative.Fill(0.0);
  DerivativeType derivativeF(this->GetNumberOfParameters(), 0.0);
  DerivativeType derivativeM(this->GetNumberOfParameters(), 0.0);
  DerivativeType differential(this->GetNumberOfParameters(), 0.0);

  /** Array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  NonZeroJacobianIndicesType nzji(Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

  /** Initialize some variables for intermediate results. */
  AccumulateType sff{};
  AccumulateType smm{};
  AccumulateType sfm{};
  AccumulateType sf{};
  AccumulateType sm{};

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

  /** Loop over the fixed image to calculate the correlation. */
  for (const auto & fixedImageSample : *sampleContainer)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = fixedImageSample.m_ImageCoordinates;
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
      Superclass::m_NumberOfPixelsCounted++;

      /** Get the fixed image value. */
      const auto fixedImageValue = static_cast<RealType>(fixedImageSample.m_ImageValue);

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the innerproducts (dM/dx)^T (dT/dmu) and (dMask/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue * movingImageValue;
      sf += fixedImageValue;
      sm += movingImageValue;

      /** Compute this pixel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji, derivativeF, derivativeM, differential);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** If NumberOfPixelsCounted > 0, then subtract things from sff, smm, sfm,
   * derivativeF and derivativeM.
   */
  const auto N = static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);
  if (Superclass::m_NumberOfPixelsCounted > 0)
  {
    sff -= (sf * sf / N);
    smm -= (sm * sm / N);
    sfm -= (sf * sm / N);

    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      derivativeF[i] -= sf * differential[i] / N;
      derivativeM[i] -= sm * differential[i] / N;
    }
  }

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * std::sqrt(sff * smm);

  /** Calculate the value and the derivative. */
  if (Superclass::m_NumberOfPixelsCounted > 0 && denom < -1e-14)
  {
    value = sfm / denom;
    for (unsigned int i = 0; i < numberOfParameters; ++i)
    {
      derivative[i] = (derivativeF[i] - (sfm / smm) * derivativeM[i]) / denom;
    }
  }
  else
  {
    value = MeasureType{};
    derivative.Fill(0.0);
  }

} // end GetValueAndDerivativeSingleThreaded()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!Superclass::m_UseMultiThread)
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

  /** launch multithreading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  this->AfterThreadedGetValueAndDerivative(value, derivative);

} // end GetValueAndDerivative()


/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(
  ThreadIdType threadId) const
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = Superclass::m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji(nnzji);
  DerivativeType               imageJacobian(nzji.size());

  /** Get handles to the pre-allocated derivatives for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & derivativeF = this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_DerivativeF;
  DerivativeType & derivativeM = this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_DerivativeM;
  DerivativeType & differential = this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Differential;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const size_t                sampleContainerSize{ sampleContainer->size() };

  /** Get the samples for this thread. */
  const auto nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto pos_begin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto pos_end = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threader_fbegin = beginOfSampleContainer + pos_begin;
  const auto threader_fend = beginOfSampleContainer + pos_end;

  /** Create variables to store intermediate results. */
  AccumulateType sff{};
  AccumulateType smm{};
  AccumulateType sfm{};
  AccumulateType sf{};
  AccumulateType sm{};
  unsigned long  numberOfPixelsCounted = 0;

  /** Loop over the fixed image to calculate the mean squares. */
  for (auto threader_fiter = threader_fbegin; threader_fiter != threader_fend; ++threader_fiter)
  {
    /** Read fixed coordinates and initialize some variables. */
    const FixedImagePointType & fixedPoint = threader_fiter->m_ImageCoordinates;
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
      const auto fixedImageValue = static_cast<RealType>(threader_fiter->m_ImageValue);

#if 0
      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian( fixedPoint, jacobian, nzji );

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(
        jacobian, movingImageDerivative, imageJacobian );
#else
      /** Compute the inner product of the transform Jacobian dT/dmu and the moving image gradient dM/dx. */
      Superclass::m_AdvancedTransform->EvaluateJacobianWithImageGradientProduct(
        fixedPoint, movingImageDerivative, imageJacobian, nzji);
#endif

      /** Update some sums needed to calculate the value of NC. */
      sff += fixedImageValue * fixedImageValue;
      smm += movingImageValue * movingImageValue;
      sfm += fixedImageValue * movingImageValue;
      sf += fixedImageValue;
      sm += movingImageValue;

      /** Compute this voxel's contribution to the derivative terms. */
      this->UpdateDerivativeTerms(
        fixedImageValue, movingImageValue, imageJacobian, nzji, derivativeF, derivativeM, differential);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Sff = sff;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Smm = smm;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Sfm = sfm;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Sf = sf;
  this->m_CorrelationGetValueAndDerivativePerThreadVariables[threadId].st_Sm = sm;

} // end ThreadedGetValueAndDerivative()


/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate the number of pixels. */
  Superclass::m_NumberOfPixelsCounted =
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_NumberOfPixelsCounted;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    Superclass::m_NumberOfPixelsCounted +=
      this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted;
  }

  /** Check if enough samples were valid. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  this->CheckNumberOfSamples();

  /** Accumulate values. */
  AccumulateType sff = this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_Sff;
  AccumulateType smm = this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_Smm;
  AccumulateType sfm = this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_Sfm;
  AccumulateType sf = this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_Sf;
  AccumulateType sm = this->m_CorrelationGetValueAndDerivativePerThreadVariables[0].st_Sm;
  for (ThreadIdType i = 1; i < numberOfThreads; ++i)
  {
    sff += this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sff;
    smm += this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Smm;
    sfm += this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sfm;
    sf += this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sf;
    sm += this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sm;

    /** Reset these variables for the next iteration. */
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sff = 0.0;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Smm = 0.0;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sfm = 0.0;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sf = 0.0;
    this->m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Sm = 0.0;
  }

  /** Subtract things from sff, smm and sfm. */
  const auto N = static_cast<RealType>(Superclass::m_NumberOfPixelsCounted);
  sff -= (sf * sf / N);
  smm -= (sm * sm / N);
  sfm -= (sf * sm / N);

  /** The denominator of the value and the derivative. */
  const RealType denom = -1.0 * std::sqrt(sff * smm);

  /** Check for sufficiently large denominator. */
  if (denom > -1e-14)
  {
    value = MeasureType{};
    derivative.Fill(0.0);
    return;
  }

  /** Calculate the metric value. */
  value = sfm / denom;

  /** Calculate the metric derivative. */
  // force multi-threaded

  MultiThreaderAccumulateDerivativeType userData;

  userData.st_Metric = const_cast<Self *>(this);
  userData.st_sf_N = sf / N;
  userData.st_sm_N = sm / N;
  userData.st_sfm_smm = sfm / smm;
  userData.st_InvertedDenominator = 1.0 / denom;
  userData.st_DerivativePointer = derivative.begin();

  this->m_Threader->SetSingleMethodAndExecute(AccumulateDerivativesThreaderCallback, &userData);

} // end AfterThreadedGetValueAndDerivative()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedNormalizedCorrelationImageToImageMetric<TFixedImage, TMovingImage>::AccumulateDerivativesThreaderCallback(
  void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadId = infoStruct.WorkUnitID;
  ThreadIdType nrOfThreads = infoStruct.NumberOfWorkUnits;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderAccumulateDerivativeType *>(infoStruct.UserData);

  assert(userData.st_Metric);
  Self & metric = *(userData.st_Metric);

  const AccumulateType sf_N = userData.st_sf_N;
  const AccumulateType sm_N = userData.st_sm_N;
  const AccumulateType sfm_smm = userData.st_sfm_smm;
  const RealType       invertedDenominator = userData.st_InvertedDenominator;

  const unsigned int numPar = metric.GetNumberOfParameters();
  const auto         subSize =
    static_cast<unsigned int>(std::ceil(static_cast<double>(numPar) / static_cast<double>(nrOfThreads)));
  const unsigned int jmin = threadId * subSize;
  const unsigned int jmax = std::min((threadId + 1) * subSize, numPar);

  for (unsigned int j = jmin; j < jmax; ++j)
  {
    DerivativeValueType derivativeF{};
    DerivativeValueType derivativeM{};
    DerivativeValueType differential{};

    for (ThreadIdType i = 0; i < nrOfThreads; ++i)
    {
      derivativeF += metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_DerivativeF[j];
      derivativeM += metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_DerivativeM[j];
      differential += metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Differential[j];

      /** Reset these variables for the next iteration. */
      metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_DerivativeF[j] = 0.0;
      metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_DerivativeM[j] = 0.0;
      metric.m_CorrelationGetValueAndDerivativePerThreadVariables[i].st_Differential[j] = 0.0;
    }

    derivativeF -= sf_N * differential;
    derivativeM -= sm_N * differential;

    userData.st_DerivativePointer[j] = (derivativeF - sfm_smm * derivativeM) * invertedDenominator;
  }

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AccumulateDerivativesThreaderCallback()


} // end namespace itk

#endif // end #ifndef _itkAdvancedNormalizedCorrelationImageToImageMetric_hxx
