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
#ifndef itkParzenWindowMutualInformationImageToImageMetric_hxx
#define itkParzenWindowMutualInformationImageToImageMetric_hxx

#include "itkParzenWindowMutualInformationImageToImageMetric.h"

#include "itkImageLinearConstIteratorWithIndex.h"
#include "itkImageScanlineConstIterator.h"
#include <vnl/vnl_math.h>
#include "itkMatrix.h"
#include <vnl/vnl_inverse.h>
#include <vnl/vnl_det.h>

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{
/**
 * ********************* Constructor ******************************
 */

template <class TFixedImage, class TMovingImage>
ParzenWindowMutualInformationImageToImageMetric<TFixedImage,
                                                TMovingImage>::ParzenWindowMutualInformationImageToImageMetric()
{
  this->m_UseJacobianPreconditioning = false;

  /** Initialize the m_ParzenWindowHistogramThreaderParameters. */
  this->m_ParzenWindowMutualInformationThreaderParameters.m_Metric = this;

} // end constructor


/**
 * ********************* InitializeHistograms ******************************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::InitializeHistograms()
{
  /** Call Superclass implementation. */
  this->Superclass::InitializeHistograms();

  /** Allocate small amount of memory for the m_PRatioArray. */
  if (!this->GetUseExplicitPDFDerivatives())
  {
    this->m_PRatioArray.SetSize(this->GetNumberOfFixedHistogramBins(), this->GetNumberOfMovingHistogramBins());
  }

} // end InitializeHistograms()


/**
 * ************************** GetValue **************************
 */

template <class TFixedImage, class TMovingImage>
auto
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
  const ParametersType & parameters) const -> MeasureType
{
  /** Construct the JointPDF and Alpha. */
  this->ComputePDFs(parameters);

  /** Normalize the pdfs: p = alpha h. */
  this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

  /** Compute the fixed and moving marginal pdfs, by summing over the joint pdf. */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  /** Compute the metric by double summation over histogram. */

  /** Setup iterators */
  using JointPDFIteratorType = ImageLinearConstIteratorWithIndex<JointPDFType>;
  using MarginalPDFIteratorType = typename MarginalPDFType::const_iterator;

  JointPDFIteratorType jointPDFit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  jointPDFit.SetDirection(0);
  jointPDFit.GoToBegin();
  MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
  const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

  /** Loop over histogram. */
  double MI = 0.0;
  while (fixedPDFit != fixedPDFend)
  {
    const double fixedImagePDFValue = *fixedPDFit;
    movingPDFit = this->m_MovingImageMarginalPDF.begin();
    while (movingPDFit != movingPDFend)
    {
      const double movingImagePDFValue = *movingPDFit;
      const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
      const double jointPDFValue = jointPDFit.Get();

      /** Check for non-zero bin contribution. */
      if (jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16)
      {
        MI += jointPDFValue * std::log(jointPDFValue / fixPDFmovPDF);
      }
      ++movingPDFit;
      ++jointPDFit;
    } // end while-loop over moving index

    ++fixedPDFit;
    jointPDFit.NextLine();

  } // end while-loop over fixed index

  return static_cast<MeasureType>(-1.0 * MI);

} // end GetValue()


/**
 * ******************** GetValueAndAnalyticDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndAnalyticDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Low memory variant. */
  if (!this->GetUseExplicitPDFDerivatives())
  {
    this->GetValueAndAnalyticDerivativeLowMemory(parameters, value, derivative);
    return;
  }

  /** Initialize some variables. */
  value = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(0.0);

  /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
  this->ComputePDFsAndPDFDerivatives(parameters);

  /** Normalize the pdfs: p = alpha h. */
  this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

  /** Compute the fixed and moving marginal pdf by summing over the histogram. */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  /** Compute the metric and derivatives by double summation over histogram. */

  /** Setup iterators .*/
  using JointPDFIteratorType = ImageLinearConstIteratorWithIndex<JointPDFType>;
  using JointPDFDerivativesIteratorType = ImageLinearConstIteratorWithIndex<JointPDFDerivativesType>;
  using MarginalPDFIteratorType = typename MarginalPDFType::const_iterator;
  using DerivativeIteratorType = typename DerivativeType::iterator;

  JointPDFIteratorType jointPDFit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  jointPDFit.SetDirection(0);
  jointPDFit.GoToBegin();
  JointPDFDerivativesIteratorType jointPDFDerivativesit(this->m_JointPDFDerivatives,
                                                        this->m_JointPDFDerivatives->GetLargestPossibleRegion());
  jointPDFDerivativesit.SetDirection(0);
  jointPDFDerivativesit.GoToBegin();
  MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
  const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();
  DerivativeIteratorType        derivit = derivative.begin();
  const DerivativeIteratorType  derivbegin = derivative.begin();
  const DerivativeIteratorType  derivend = derivative.end();

  /** Loop over the joint histogram. */
  double MI = 0.0;
  while (fixedPDFit != fixedPDFend)
  {
    const double fixedImagePDFValue = *fixedPDFit;
    movingPDFit = this->m_MovingImageMarginalPDF.begin();
    while (movingPDFit != movingPDFend)
    {
      const double movingImagePDFValue = *movingPDFit;
      const double fixPDFmovPDF = fixedImagePDFValue * movingImagePDFValue;
      const double jointPDFValue = jointPDFit.Get();

      /** Check for non-zero bin contribution. */
      if (jointPDFValue > 1e-16 && fixPDFmovPDF > 1e-16)
      {
        derivit = derivbegin;
        const double pRatio = std::log(jointPDFValue / fixPDFmovPDF);
        const double pRatioAlpha = this->m_Alpha * pRatio;
        MI += jointPDFValue * pRatio;
        while (derivit != derivend)
        {
          /**  Ref: eq 23 of Thevenaz & Unser paper [3]. */
          (*derivit) -= jointPDFDerivativesit.Get() * pRatioAlpha;
          ++derivit;
          ++jointPDFDerivativesit;
        } // end while-loop over parameters
      }   // end if-block to check non-zero bin contribution

      ++movingPDFit;
      ++jointPDFit;
      jointPDFDerivativesit.NextLine();

    } // end while-loop over moving index
    ++fixedPDFit;
    jointPDFit.NextLine();
  } // end while-loop over fixed index

  value = static_cast<MeasureType>(-1.0 * MI);

} // end GetValueAndAnalyticDerivative()


/**
 * ******************** GetValueAndAnalyticDerivativeLowMemory *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndAnalyticDerivativeLowMemory(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Construct the JointPDF and Alpha.
   * This function contains a loop over the samples.
   * It executes multi-threadedly when m_UseMultiThread == true.
   */
  this->ComputePDFs(parameters);

  /** Normalize the joint histogram by alpha. */
  this->NormalizeJointPDF(this->m_JointPDF, this->m_Alpha);

  /** Compute the fixed and moving marginal pdf by summing over the histogram. */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  // \todo: the last three loops over the joint histogram can be done in
  // one loop, maybe also include the next loop to generate m_PRatioArray.
  // The effort is probably not worth the gain in performance.

  /** Compute the metric value and the intermediate m_PRatioArray
   * by summation over the joint histogram.
   */
  double MI = 0.0;
  this->ComputeValueAndPRatioArray(MI);
  value = static_cast<MeasureType>(-1.0 * MI);

  /* Compute the derivative.
   * This function contains a second loop over the samples.
   * It executes multi-threadedly when m_UseMultiThread == true.
   */
  this->ComputeDerivativeLowMemory(derivative);

} // end GetValueAndAnalyticDerivativeLowMemory()


/**
 * ******************** ComputeDerivativeLowMemorySingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeDerivativeLowMemorySingleThreaded(
  DerivativeType & derivative) const
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji = NonZeroJacobianIndicesType(nnzji);
  DerivativeType               imageJacobian(nzji.size());
  TransformJacobianType        jacobian;
  derivative.Fill(0.0);

  /** Declare and allocate arrays for Jacobian preconditioning. */
  DerivativeType jacobianPreconditioner, preconditioningDivisor;
  if (this->GetUseJacobianPreconditioning())
  {
    jacobianPreconditioner = DerivativeType(nzji.size());
    preconditioningDivisor = DerivativeType(this->GetNumberOfParameters());
    preconditioningDivisor.Fill(0.0);
  }

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->End();

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and create some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value, its derivative, and check
     * if the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk =
        this->Superclass::EvaluateMovingImageValueAndDerivative(mappedPoint, movingImageValue, &movingImageDerivative);
    }

    if (sampleOk)
    {
      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

      /** Make sure the values fall within the histogram range. */
      fixedImageValue = this->GetFixedImageLimiter()->Evaluate(fixedImageValue);
      movingImageValue = this->GetMovingImageLimiter()->Evaluate(movingImageValue, movingImageDerivative);

      /** Get the transform Jacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the inner product (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** If desired, apply the technique introduced by Tustison. */
      if (this->GetUseJacobianPreconditioning())
      {
        this->ComputeJacobianPreconditioner(jacobian, nzji, jacobianPreconditioner, preconditioningDivisor);
        DerivativeValueType * imjacit = imageJacobian.begin();
        DerivativeValueType * jacprecit = jacobianPreconditioner.begin();
        for (unsigned int i = 0; i < nzji.size(); ++i)
        {
          while (imjacit != imageJacobian.end())
          {
            (*imjacit) *= (*jacprecit);
            ++imjacit;
            ++jacprecit;
          }
        }
      }

      /** Compute this sample's contribution to the joint distributions. */
      this->UpdateDerivativeLowMemory(fixedImageValue, movingImageValue, imageJacobian, nzji, derivative);

    } // end sampleOk
  }   // end loop over sample container

  /** If desired, apply the technique introduced by Tustison */
  if (this->GetUseJacobianPreconditioning())
  {
    DerivativeValueType * derivit = derivative.begin();
    DerivativeValueType * divisit = preconditioningDivisor.begin();

    /** This normalization was not in the Tustison paper, but it helps,
     * especially for localized mutual information.
     */
    const double normalizationFactor = preconditioningDivisor.mean();
    while (derivit != derivative.end())
    {
      (*derivit) *= normalizationFactor / ((*divisit) + 1e-14);
      ++derivit;
      ++divisit;
    }
  }

} // end ComputeDerivativeLowMemorySingleThreaded()


/**
 * ******************** ComputeDerivativeLowMemory *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeDerivativeLowMemory(
  DerivativeType & derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->ComputeDerivativeLowMemorySingleThreaded(derivative);
  }

  /** Launch multi-threading derivative computation. */
  this->LaunchComputeDerivativeLowMemoryThreaderCallback();

  /** Gather the results from all threads. */
  this->AfterThreadedComputeDerivativeLowMemory(derivative);

} // end ComputeDerivativeLowMemory()


/**
 * ******************* ThreadedComputeDerivativeLowMemory *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ThreadedComputeDerivativeLowMemory(
  ThreadIdType threadId)
{
  /** Initialize array that stores dM(x)/dmu, and the sparse Jacobian + indices. */
  const NumberOfParametersType nnzji = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   nzji = NonZeroJacobianIndicesType(nnzji);
  DerivativeType               imageJacobian(nzji.size());

  /** Get a handle to the pre-allocated derivative for the current thread.
   * The initialization is performed at the beginning of each resolution in
   * InitializeThreadingParameters(), and at the end of each iteration in
   * AfterThreadedGetValueAndDerivative() and the accumulate functions.
   */
  DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Derivative;

  /** Declare and allocate arrays for Jacobian preconditioning. */
  DerivativeType jacobianPreconditioner, preconditioningDivisor;
  if (this->GetUseJacobianPreconditioning())
  {
    jacobianPreconditioner = DerivativeType(nzji.size());
    preconditioningDivisor = DerivativeType(this->GetNumberOfParameters());
    preconditioningDivisor.Fill(0.0);
  }

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
  typename ImageSampleContainerType::ConstIterator fiter;
  typename ImageSampleContainerType::ConstIterator fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator fend = sampleContainer->Begin();
  fbegin += (int)pos_begin;
  fend += (int)pos_end;

  /** Loop over sample container and compute contribution of each sample to pdfs. */
  for (fiter = fbegin; fiter != fend; ++fiter)
  {
    /** Read fixed coordinates and create some variables. */
    const FixedImagePointType & fixedPoint = fiter->Value().m_ImageCoordinates;
    RealType                    movingImageValue;
    MovingImageDerivativeType   movingImageDerivative;

    /** Transform point. */
    const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);

    /** Check if the point is inside the moving mask. */
    bool sampleOk = this->IsInsideMovingMask(mappedPoint);

    /** Compute the moving image value, its derivative, and check
     * if the point is inside the moving image buffer.
     */
    if (sampleOk)
    {
      sampleOk = this->FastEvaluateMovingImageValueAndDerivative(
        mappedPoint, movingImageValue, &movingImageDerivative, threadId);
    }

    if (sampleOk)
    {
      /** Get the fixed image value. */
      RealType fixedImageValue = static_cast<RealType>(fiter->Value().m_ImageValue);

      /** Make sure the values fall within the histogram range. */
      fixedImageValue = this->GetFixedImageLimiter()->Evaluate(fixedImageValue);
      movingImageValue = this->GetMovingImageLimiter()->Evaluate(movingImageValue, movingImageDerivative);

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

      /** If desired, apply the technique introduced by Tustison. */
      TransformJacobianType jacobian;
      if (this->GetUseJacobianPreconditioning())
      {
        this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

        this->ComputeJacobianPreconditioner(jacobian, nzji, jacobianPreconditioner, preconditioningDivisor);
        DerivativeValueType * imjacit = imageJacobian.begin();
        DerivativeValueType * jacprecit = jacobianPreconditioner.begin();
        for (unsigned int i = 0; i < nzji.size(); ++i)
        {
          while (imjacit != imageJacobian.end())
          {
            (*imjacit) *= (*jacprecit);
            ++imjacit;
            ++jacprecit;
          }
        }
      }

      /** Compute this sample's contribution to the joint distributions. */
      this->UpdateDerivativeLowMemory(fixedImageValue, movingImageValue, imageJacobian, nzji, derivative);

    } // end sampleOk
  }   // end loop over sample container

  /** If desired, apply the technique introduced by Tustison. */
  if (this->GetUseJacobianPreconditioning())
  {
    DerivativeValueType * derivit = derivative.begin();
    DerivativeValueType * divisit = preconditioningDivisor.begin();

    /** This normalization was not in the Tustison paper, but it helps,
     * especially for localized mutual information.
     */
    const double normalizationFactor = preconditioningDivisor.mean();
    while (derivit != derivative.end())
    {
      (*derivit) *= normalizationFactor / ((*divisit) + 1e-14);
      ++derivit;
      ++divisit;
    }
  }

} // end ThreadedComputeDerivativeLowMemory()


/**
 * ******************* AfterThreadedComputeDerivativeLowMemory *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedComputeDerivativeLowMemory(
  DerivativeType & derivative) const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Accumulate derivatives. */
  // compute single-threadedly
  if (!this->m_UseMultiThread && false) // force multi-threaded
  {
    derivative = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      derivative += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative;
    }
  }
#ifdef ELASTIX_USE_OPENMP
  // compute multi-threadedly with openmp
  else if (false) // this->m_UseOpenMP )
  {
    const int spaceDimension = static_cast<int>(this->GetNumberOfParameters());

#  pragma omp parallel for
    for (int j = 0; j < spaceDimension; ++j)
    {
      DerivativeValueType sum = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative[j];
      for (ThreadIdType i = 1; i < numberOfThreads; ++i)
      {
        sum += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j];
      }
      derivative[j] = sum;
    }
  }
#endif
  // compute multi-threadedly with itk threads
  else
  {
    this->m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
    this->m_ThreaderMetricParameters.st_NormalizationFactor = 1.0;

    this->m_Threader->SetSingleMethod(this->AccumulateDerivativesThreaderCallback,
                                      const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));
    this->m_Threader->SingleMethodExecute();
  }

} // end AfterThreadedComputeDerivativeLowMemory()


/**
 * **************** ComputeDerivativeLowMemoryThreaderCallback *******
 */

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeDerivativeLowMemoryThreaderCallback(
  void * arg)
{
  ThreadInfoType * infoStruct = static_cast<ThreadInfoType *>(arg);
  ThreadIdType     threadId = infoStruct->WorkUnitID;

  ParzenWindowMutualInformationMultiThreaderParameterType * temp =
    static_cast<ParzenWindowMutualInformationMultiThreaderParameterType *>(infoStruct->UserData);

  temp->m_Metric->ThreadedComputeDerivativeLowMemory(threadId);

  return itk::ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end ComputeDerivativeLowMemoryThreaderCallback()


/**
 * *********************** LaunchComputeDerivativeLowMemoryThreaderCallback***************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage,
                                                TMovingImage>::LaunchComputeDerivativeLowMemoryThreaderCallback() const
{
  /** Setup threader. */
  this->m_Threader->SetSingleMethod(
    this->ComputeDerivativeLowMemoryThreaderCallback,
    const_cast<void *>(static_cast<const void *>(&this->m_ParzenWindowMutualInformationThreaderParameters)));

  /** Launch. */
  this->m_Threader->SingleMethodExecute();

} // end LaunchComputeDerivativeLowMemoryThreaderCallback()


/**
 * ******************* ComputeValueAndPRatioArray *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndPRatioArray(
  double & MI) const
{
  /** Setup iterators. */
  using JointPDFIteratorType = ImageScanlineConstIterator<JointPDFType>;
  using MarginalPDFIteratorType = typename MarginalPDFType::const_iterator;

  JointPDFIteratorType          jointPDFit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
  const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  MarginalPDFIteratorType       movingPDFit;
  const MarginalPDFIteratorType movingPDFbegin = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

  /** Initialize */
  this->m_PRatioArray.Fill(itk::NumericTraits<PRatioType>::ZeroValue());

  /** Loop over the joint histogram. */
  PDFValueType sum = 0.0;
  unsigned int fixedIndex = 0;
  unsigned int movingIndex = 0;
  while (fixedPDFit != fixedPDFend)
  {
    const double fixedPDFValue = *fixedPDFit;
    double       logFixedPDFValue = 0.0;
    if (fixedPDFValue > 1e-16)
    {
      logFixedPDFValue = std::log(fixedPDFValue);
    }
    movingPDFit = movingPDFbegin;
    movingIndex = 0;

    while (movingPDFit != movingPDFend)
    {
      const PDFValueType movingPDFValue = *movingPDFit;
      const PDFValueType jointPDFValue = jointPDFit.Value();

      /** Check for non-zero bin contribution. */
      if (jointPDFValue > 1e-16 && movingPDFValue > 1e-16)
      {
        const PDFValueType pRatio = std::log(jointPDFValue / movingPDFValue);
        this->m_PRatioArray[fixedIndex][movingIndex] = static_cast<PRatioType>(this->m_Alpha * pRatio);

        if (fixedPDFValue > 1e-16)
        {
          sum += jointPDFValue * (pRatio - logFixedPDFValue);
        }
      } // end if-block to check non-zero bin contribution

      /** Update iterators. */
      ++movingPDFit;
      ++jointPDFit;
      ++movingIndex;

    } // end while-loop over moving index

    /** Update iterators. */
    ++fixedPDFit;
    jointPDFit.NextLine();
    ++fixedIndex;

  } // end while-loop over fixed index

  // Assign
  MI = sum;

} // end ComputeValueAndPRatioArray()


/**
 * ******************* UpdateDerivativeLowMemory *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::UpdateDerivativeLowMemory(
  const RealType &                   fixedImageValue,
  const RealType &                   movingImageValue,
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType &                   derivative) const
{
  /** In this function we need to do (see eq. 24 of Thevenaz [3]):
   *      derivative -= constant * imageJacobian *
   *          \sum_i \sum_k PRatio(i,k) * dB/dxi(xi,i,k),
   * with i, k, the fixed and moving histogram bins,
   * PRatio the precomputed log( p(i,k) / p(i) ), and
   * dB/dxi the B-spline derivative.
   *
   * Note (1) that we only have to loop over i,k within the support
   * of the B-spline Parzen-window.
   * Note (2) that imageJacobian may be sparse.
   */

  /** Determine the affected region. */

  /** Determine Parzen window arguments (see eq. 6 of Mattes paper [2]). */
  const double fixedImageParzenWindowTerm =
    fixedImageValue / this->m_FixedImageBinSize - this->m_FixedImageNormalizedMin;
  const double movingImageParzenWindowTerm =
    movingImageValue / this->m_MovingImageBinSize - this->m_MovingImageNormalizedMin;

  /** The lowest bin numbers affected by this pixel: */
  const int fixedParzenWindowIndex =
    static_cast<int>(std::floor(fixedImageParzenWindowTerm + this->m_FixedParzenTermToIndexOffset));
  const int movingParzenWindowIndex =
    static_cast<int>(std::floor(movingImageParzenWindowTerm + this->m_MovingParzenTermToIndexOffset));

  /** Compute the fixed Parzen values. */
  ParzenValueContainerType fixedParzenValues(this->m_JointPDFWindow.GetSize()[1]);
  this->EvaluateParzenValues(
    fixedImageParzenWindowTerm, fixedParzenWindowIndex, this->m_FixedKernel, fixedParzenValues);

  /** Compute the derivatives of the moving Parzen window. */
  ParzenValueContainerType derivativeMovingParzenValues(this->m_JointPDFWindow.GetSize()[0]);
  this->EvaluateParzenValues(
    movingImageParzenWindowTerm, movingParzenWindowIndex, this->m_DerivativeMovingKernel, derivativeMovingParzenValues);

  /** Get the moving image bin size. */
  const double et = static_cast<double>(this->m_MovingImageBinSize);

  /** Loop over the Parzen window region and increment sum. */
  PDFValueType sum = 0.0;
  for (unsigned int f = 0; f < fixedParzenValues.GetSize(); ++f)
  {
    const double fv_et = fixedParzenValues[f] / et;
    for (unsigned int m = 0; m < derivativeMovingParzenValues.GetSize(); ++m)
    {
      sum += this->m_PRatioArray[f + fixedParzenWindowIndex][m + movingParzenWindowIndex] * fv_et *
             derivativeMovingParzenValues[m];
    }
  }

  const auto numberOfParameters = this->GetNumberOfParameters();

  /** Now compute derivative -= sum * imageJacobian. */
  if (nzji.size() == numberOfParameters)
  {
    /** Loop over all Jacobians. */
    for (unsigned int mu = 0; mu < numberOfParameters; ++mu)
    {
      derivative[mu] += static_cast<DerivativeValueType>(imageJacobian[mu] * sum);
    }
  }
  else
  {
    /** Loop only over the non-zero Jacobians. */
    for (unsigned int i = 0; i < imageJacobian.GetSize(); ++i)
    {
      const unsigned int mu = nzji[i];
      derivative[mu] += static_cast<DerivativeValueType>(imageJacobian[i] * sum);
    }
  }

} // end UpdateDerivativeLowMemory()


/**
 * ******************** GetValueAndFiniteDifferenceDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndFiniteDifferenceDerivative(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{
  /** Initialize some variables. */
  value = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(0.0);

  /** Construct the JointPDF, JointPDFDerivatives, Alpha and its derivatives. */
  this->ComputePDFsAndIncrementalPDFs(parameters);

  /** Compute the fixed and moving marginal pdf by summing over the histogram. */
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_FixedImageMarginalPDF, 0);
  this->ComputeMarginalPDF(this->m_JointPDF, this->m_MovingImageMarginalPDF, 1);

  /** Compute the fixed and moving incremental marginal pdfs by summing over the
   * incremental histogram. Do it for Right and Left.
   */
  this->ComputeIncrementalMarginalPDFs(this->m_IncrementalJointPDFRight,
                                       this->m_FixedIncrementalMarginalPDFRight,
                                       this->m_MovingIncrementalMarginalPDFRight);
  this->ComputeIncrementalMarginalPDFs(
    this->m_IncrementalJointPDFLeft, this->m_FixedIncrementalMarginalPDFLeft, this->m_MovingIncrementalMarginalPDFLeft);

  /** Compute the metric and derivatives by double summation over histogram. */

  /** Setup iterators */
  using JointPDFIteratorType = ImageLinearConstIteratorWithIndex<JointPDFType>;
  using IncrementalJointPDFIteratorType = ImageLinearConstIteratorWithIndex<JointPDFDerivativesType>;
  using MarginalPDFIteratorType = typename MarginalPDFType::const_iterator;
  using IncrementalMarginalPDFIteratorType = ImageLinearConstIteratorWithIndex<IncrementalMarginalPDFType>;
  using DerivativeIteratorType = typename DerivativeType::iterator;
  using DerivativeConstIteratorType = typename DerivativeType::const_iterator;

  JointPDFIteratorType jointPDFit(this->m_JointPDF, this->m_JointPDF->GetLargestPossibleRegion());
  jointPDFit.GoToBegin();

  IncrementalJointPDFIteratorType jointIncPDFRightit(this->m_IncrementalJointPDFRight,
                                                     this->m_IncrementalJointPDFRight->GetLargestPossibleRegion());
  IncrementalJointPDFIteratorType jointIncPDFLeftit(this->m_IncrementalJointPDFLeft,
                                                    this->m_IncrementalJointPDFLeft->GetLargestPossibleRegion());
  jointIncPDFRightit.GoToBegin();
  jointIncPDFLeftit.GoToBegin();

  MarginalPDFIteratorType       fixedPDFit = this->m_FixedImageMarginalPDF.begin();
  const MarginalPDFIteratorType fixedPDFend = this->m_FixedImageMarginalPDF.end();
  MarginalPDFIteratorType       movingPDFit = this->m_MovingImageMarginalPDF.begin();
  const MarginalPDFIteratorType movingPDFend = this->m_MovingImageMarginalPDF.end();

  IncrementalMarginalPDFIteratorType fixedIncPDFRightit(
    this->m_FixedIncrementalMarginalPDFRight, this->m_FixedIncrementalMarginalPDFRight->GetLargestPossibleRegion());
  IncrementalMarginalPDFIteratorType movingIncPDFRightit(
    this->m_MovingIncrementalMarginalPDFRight, this->m_MovingIncrementalMarginalPDFRight->GetLargestPossibleRegion());
  IncrementalMarginalPDFIteratorType fixedIncPDFLeftit(
    this->m_FixedIncrementalMarginalPDFLeft, this->m_FixedIncrementalMarginalPDFLeft->GetLargestPossibleRegion());
  IncrementalMarginalPDFIteratorType movingIncPDFLeftit(
    this->m_MovingIncrementalMarginalPDFLeft, this->m_MovingIncrementalMarginalPDFLeft->GetLargestPossibleRegion());
  fixedIncPDFRightit.GoToBegin();
  movingIncPDFRightit.GoToBegin();
  fixedIncPDFLeftit.GoToBegin();
  movingIncPDFLeftit.GoToBegin();

  DerivativeIteratorType       derivit = derivative.begin();
  const DerivativeIteratorType derivbegin = derivative.begin();
  const DerivativeIteratorType derivend = derivative.end();

  DerivativeConstIteratorType       perturbedAlphaRightit = this->m_PerturbedAlphaRight.begin();
  const DerivativeConstIteratorType perturbedAlphaRightbegin = this->m_PerturbedAlphaRight.begin();
  DerivativeConstIteratorType       perturbedAlphaLeftit = this->m_PerturbedAlphaLeft.begin();
  const DerivativeConstIteratorType perturbedAlphaLeftbegin = this->m_PerturbedAlphaLeft.begin();

  double MI = 0.0;
  while (fixedPDFit != fixedPDFend)
  {
    const double fixedPDFValue = *fixedPDFit;

    while (movingPDFit != movingPDFend)
    {
      const double movingPDFValue = *movingPDFit;
      const double jointPDFValue = jointPDFit.Get();
      const double fixPDFmovPDFAlpha = fixedPDFValue * movingPDFValue * this->m_Alpha;

      /** Check for non-zero bin contribution and update the mutual information value. */
      if (jointPDFValue > 1e-16 && fixPDFmovPDFAlpha > 1e-16)
      {
        MI += this->m_Alpha * jointPDFValue * std::log(jointPDFValue / fixPDFmovPDFAlpha);
      }

      /** Update the derivative. */
      derivit = derivbegin;
      perturbedAlphaRightit = perturbedAlphaRightbegin;
      perturbedAlphaLeftit = perturbedAlphaLeftbegin;
      while (derivit != derivend)
      {
        /** Initialize. */
        double contrib = 0.0;

        /** For clarity, get some values and give them a name.
         * \todo Does this cost a lot of computation time?
         */
        const double jointIncPDFRightValue = jointIncPDFRightit.Get();
        const double fixedIncPDFRightValue = fixedIncPDFRightit.Get();
        const double movingIncPDFRightValue = movingIncPDFRightit.Get();
        const double perturbedAlphaRightValue = *perturbedAlphaRightit;

        /** Compute the contribution of the Right-perturbation to the derivative. */
        const double perturbedJointPDFRightValue = jointIncPDFRightValue + jointPDFValue;
        const double perturbedFixedPDFRightValue = fixedPDFValue + fixedIncPDFRightValue;
        const double perturbedMovingPDFRightValue = movingPDFValue + movingIncPDFRightValue;
        const double perturbedfixPDFmovPDFAlphaRight =
          perturbedFixedPDFRightValue * perturbedMovingPDFRightValue * perturbedAlphaRightValue;

        if (perturbedJointPDFRightValue > 1e-16 && perturbedfixPDFmovPDFAlphaRight > 1e-16)
        {
          contrib += perturbedAlphaRightValue * perturbedJointPDFRightValue *
                     std::log(perturbedJointPDFRightValue / perturbedfixPDFmovPDFAlphaRight);
        }

        /** For clarity, get some values and give them a name. */
        const double jointIncPDFLeftValue = jointIncPDFLeftit.Get();
        const double fixedIncPDFLeftValue = fixedIncPDFLeftit.Get();
        const double movingIncPDFLeftValue = movingIncPDFLeftit.Get();
        const double perturbedAlphaLeftValue = *perturbedAlphaLeftit;

        /** Compute the contribution of the Left-perturbation to the derivative. */
        const double perturbedJointPDFLeftValue = jointIncPDFLeftValue + jointPDFValue;
        const double perturbedFixedPDFLeftValue = fixedPDFValue + fixedIncPDFLeftValue;
        const double perturbedMovingPDFLeftValue = movingPDFValue + movingIncPDFLeftValue;
        const double perturbedfixPDFmovPDFAlphaLeft =
          perturbedFixedPDFLeftValue * perturbedMovingPDFLeftValue * perturbedAlphaLeftValue;

        if (perturbedJointPDFLeftValue > 1e-16 && perturbedfixPDFmovPDFAlphaLeft > 1e-16)
        {
          contrib -= perturbedAlphaLeftValue * perturbedJointPDFLeftValue *
                     std::log(perturbedJointPDFLeftValue / perturbedfixPDFmovPDFAlphaLeft);
        }

        /** Update the derivative component. */
        (*derivit) += contrib;

        /** Move the iterators to the next parameter. */
        ++derivit;
        ++perturbedAlphaRightit;
        ++perturbedAlphaLeftit;
        ++jointIncPDFRightit;
        ++jointIncPDFLeftit;
        ++fixedIncPDFRightit;
        ++movingIncPDFRightit;
        ++fixedIncPDFLeftit;
        ++movingIncPDFLeftit;
      } // end while-loop over parameters

      ++jointPDFit;                         // next moving bin
      ++movingPDFit;                        // next moving bin
      jointIncPDFRightit.NextLine();        // next moving bin
      jointIncPDFLeftit.NextLine();         // next moving bin
      fixedIncPDFRightit.GoToBeginOfLine(); // same fixed bin
      fixedIncPDFLeftit.GoToBeginOfLine();  // same fixed bin
      movingIncPDFRightit.NextLine();       // next moving bin
      movingIncPDFLeftit.NextLine();        // next moving bin

    } // end while-loop over moving index

    jointPDFit.NextLine();                                // next fixed bin
    ++fixedPDFit;                                         // next fixed bin
    movingPDFit = this->m_MovingImageMarginalPDF.begin(); // first moving bin
    fixedIncPDFRightit.NextLine();                        // next fixed bin
    fixedIncPDFLeftit.NextLine();                         // next fixed bin
    movingIncPDFRightit.GoToBegin();                      // first moving bin
    movingIncPDFLeftit.GoToBegin();                       // first moving bin

  } // end while-loop over fixed index

  value = static_cast<MeasureType>(-1.0 * MI);

  /** Divide the derivative by -delta*2. */
  const double delta2 = -1.0 / (this->GetFiniteDifferencePerturbation() * 2.0);
  derivit = derivative.begin();
  while (derivit != derivend)
  {
    (*derivit) *= delta2;
    ++derivit;
  }

} // end GetValueAndFiniteDifferenceDerivative


/**
 * ******************** ComputeJacobianPreconditioner *******************
 */

template <class TFixedImage, class TMovingImage>
void
ParzenWindowMutualInformationImageToImageMetric<TFixedImage, TMovingImage>::ComputeJacobianPreconditioner(
  const TransformJacobianType &      jac,
  const NonZeroJacobianIndicesType & nzji,
  DerivativeType &                   preconditioner,
  DerivativeType &                   divisor) const
{
  using TransformJacobianValueType = typename TransformJacobianType::ValueType;
  const unsigned int M = nzji.size();
  using MatrixType = Matrix<double, MovingImageDimension, MovingImageDimension>;
  MatrixType jacjact;

  /** Compute jac * jac' */
  for (unsigned int drow = 0; drow < MovingImageDimension; ++drow)
  {
    for (unsigned int dcol = drow; dcol < MovingImageDimension; ++dcol)
    {
      const TransformJacobianValueType * jacit1 = jac[drow];
      const TransformJacobianValueType * jacit2 = jac[dcol];
      double                             sum = 0.0;
      for (unsigned int mu = 0; mu < M; ++mu)
      {
        sum += (*jacit1) * (*jacit2);
        ++jacit1;
        ++jacit2;
      }
      jacjact(drow, dcol) = sum;
      jacjact(dcol, drow) = sum;
    }
  }

  /** Invert */
  const double addtodiag = 1e-10;
  for (unsigned int drow = 0; drow < MovingImageDimension; ++drow)
  {
    jacjact(drow, drow) += addtodiag;
  }
  jacjact = vnl_inverse(jacjact.GetVnlMatrix());

  /** Compute preconditioner = diag( jac' * m * jac ),
   * with m = inv(jacjact)
   * implementation:
   * preconditioner = sum_dr sum_dc m(dr,dc) jac(dr,:) * jac(dc,:)
   */
  preconditioner.Fill(0.0);
  for (unsigned int drow = 0; drow < MovingImageDimension; ++drow)
  {
    for (unsigned int dcol = drow; dcol < MovingImageDimension; ++dcol)
    {
      DerivativeValueType *              precondit = preconditioner.begin();
      const TransformJacobianValueType * jacit1 = jac[drow];
      const TransformJacobianValueType * jacit2 = jac[dcol];
      /** count twice if off-diagonal */
      const double fac = drow == dcol ? 1.0 : 2.0;
      const double m = fac * jacjact(drow, dcol);
      for (unsigned int mu = 0; mu < M; ++mu)
      {
        *precondit += m * (*jacit1) * (*jacit2);
        ++precondit;
        ++jacit1;
        ++jacit2;
      }
    }
  }

  /** Update divisor = sum_samples diag(jac'*jac) */
  DerivativeType temp(M);
  temp.Fill(0.0);
  /** Compute this sample's contribution */
  for (unsigned int drow = 0; drow < MovingImageDimension; ++drow)
  {
    DerivativeValueType *              tempit = temp.begin();
    const TransformJacobianValueType * jacit1 = jac[drow];
    for (unsigned int mu = 0; mu < M; ++mu)
    {
      *tempit += vnl_math::sqr(*jacit1);
      ++tempit;
      ++jacit1;
    }
  }
  /** Update divisor */
  for (unsigned int mu = 0; mu < M; ++mu)
  {
    divisor[nzji[mu]] += temp[mu];
  }

} // end ComputeJacobianPreconditioner()


} // end namespace itk

#endif // end #ifndef itkParzenWindowMutualInformationImageToImageMetric_hxx
