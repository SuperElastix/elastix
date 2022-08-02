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
#ifndef _itkSumSquaredTissueVolumeDifferenceImageToImageMetric_txx
#define _itkSumSquaredTissueVolumeDifferenceImageToImageMetric_txx

#include "itkSumSquaredTissueVolumeDifferenceImageToImageMetric.h"
#include <vnl/algo/vnl_matrix_update.h>

#ifdef ELASTIX_USE_OPENMP
#  include <omp.h>
#endif

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedImage, class TMovingImage>
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage,
                                                   TMovingImage>::SumSquaredTissueVolumeDifferenceImageToImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);
  this->m_AirValue = -1000.0;
  this->m_TissueValue = 55.0;

} // end Constructor

/**
 * ******************* PrintSelf *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os,
                                                                                         Indent         indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "AirValue: " << this->m_AirValue << std::endl;
  os << indent << "TissueValue: " << this->m_TissueValue << std::endl;

} // end PrintSelf()


/**
 * ******************* GetValueSingleThreaded *******************
 */

template <class TFixedImage, class TMovingImage>
auto
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValueSingleThreaded(
  const TransformParametersType & parameters) const -> MeasureType
{
  itkDebugMacro("GetValue( " << parameters << " ) ");

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;

  /** Matrix to store the spatial Jacobian, dT/dx. */
  SpatialJacobianType spatialJac;

  /** Make sure the transform parameters are up to date. */
  /** Update the imageSampler.  */
  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** and get a handle to the sample container. */
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

      /** Get the SpatialJacobian dT/dx. */
      this->m_AdvancedTransform->GetSpatialJacobian(fixedPoint, spatialJac);

      /** Compute the determinant of the Transform Jacobian |dT/dx|. */
      const RealType detjac = static_cast<RealType>(vnl_det(spatialJac.GetVnlMatrix()));

      /** Get the fixed image value. */
      const RealType & fixedImageValue = static_cast<double>(fiter->Value().m_ImageValue);

      /** The difference squared. */
      const RealType diff = ((fixedImageValue - this->m_AirValue) - detjac * (movingImageValue - this->m_AirValue)) /
                            (this->m_TissueValue - this->m_AirValue);
      measure += diff * diff;

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Update measure value. */
  double sum = 0.0;
  if (this->m_NumberOfPixelsCounted > 0)
  {
    sum = 1.0F / static_cast<double>(this->m_NumberOfPixelsCounted);
  }
  measure *= sum;

  /** Return the mean squares measure value. */
  return measure;

} // end GetValueSingleThreaded()


/**
 * ******************* GetValue *******************
 */

template <class TFixedImage, class TMovingImage>
auto
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValue(
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
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValue(ThreadIdType threadId)
{
  /*Create variables to store intermediate results. Circumvent false sharing*/
  unsigned long numberOfPixelsCounted = 0;
  MeasureType   measure = NumericTraits<MeasureType>::Zero;

  /** Matrix to store the spatial Jacobian, dT/dx. */
  SpatialJacobianType spatialJac;

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nSamplesPerThread = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  unsigned long pos_begin = nSamplesPerThread * threadId;
  unsigned long pos_end = nSamplesPerThread * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();

  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

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

      /** Get the SpatialJacobian dT/dx. */
      this->m_AdvancedTransform->GetSpatialJacobian(fixedPoint, spatialJac);

      /** Compute the determinant of the Transform Jacobian |dT/dx|. */
      const RealType detjac = static_cast<RealType>(vnl_det(spatialJac.GetVnlMatrix()));

      /** The difference squared. */
      const RealType diff = ((fixedImageValue - this->m_AirValue) - detjac * (movingImageValue - this->m_AirValue)) /
                            (this->m_TissueValue - this->m_AirValue);
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
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValue(
  MeasureType & value) const
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

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }
  value /= static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

} // end AfterThreadedGetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(
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


/** Get value and derivatives single-threaded */
template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  itkDebugMacro("GetValueAndDerivative( " << parameters << " ) ");

  /** Initialize some variables. */
  this->m_NumberOfPixelsCounted = 0;
  MeasureType measure = NumericTraits<MeasureType>::Zero;
  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::Zero);

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

  /** Matrix to store the spatial Jacobian, dT/dx. */
  SpatialJacobianType spatialJac;

  /** Matrix to store the scaled inverse spatial Jacobian, det(dT/dx) * (dT/dx)^-1 */
  SpatialJacobianType inverseSpatialJacobian;

  /** Array that stores JacobianOfSpatialJacobian, d(dT/dx)/dmu */
  JacobianOfSpatialJacobianType jacobianOfSpatialJacobian;

  DerivativeType jacobianOfSpatialJacobianDeterminant(nzji.size());

  /** Make sure the transform parameters are up to date. */
  /** Update the imageSampler. */
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

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Get the SpatialJacobian dT/dx. */
      this->m_AdvancedTransform->GetSpatialJacobian(fixedPoint, spatialJac);

      /** Compute the determinant of the Transform Jacobian |dT/dx|. */
      const RealType detjac = static_cast<RealType>(vnl_det(spatialJac.GetVnlMatrix()));

      /** Compute the inverse spatialJacobian. */
      inverseSpatialJacobian = spatialJac.GetInverse();

      /** Compute the JacobianOfSpatialJacobian. */
      this->m_AdvancedTransform->GetJacobianOfSpatialJacobian(fixedPoint, jacobianOfSpatialJacobian, nzji);

      /** Compute the dot product of the inverse spatialJacobian and JacobianOfSpatialJacobian
       * to support calculation of the JacobianOfSpatialJacobianDeterminant. */
      this->EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct(
        jacobianOfSpatialJacobian, inverseSpatialJacobian, jacobianOfSpatialJacobianDeterminant);

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(fixedImageValue,
                                          movingImageValue,
                                          imageJacobian,
                                          nzji,
                                          detjac,
                                          jacobianOfSpatialJacobianDeterminant,
                                          measure,
                                          derivative);

    } // end if sampleOk

  } // end for loop over the image sample container

  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples(sampleContainer->Size(), this->m_NumberOfPixelsCounted);

  /** Compute the measure value and derivative. */
  double sum = 0.0;
  if (this->m_NumberOfPixelsCounted > 0)
  {
    sum = 1.0F / static_cast<double>(this->m_NumberOfPixelsCounted);
  }
  measure *= sum;
  derivative *= sum;

  /** The return value. */
  value = measure;
}


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(
  const TransformParametersType & parameters,
  MeasureType &                   value,
  DerivativeType &                derivative) const
{
  /** Option for now to still use the single threaded code. */
  if (!this->m_UseMultiThread)
  {
    return this->GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
  }

  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Initialize some threading related parameters. */
  this->InitializeThreadingParameters();

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
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(
  ThreadIdType threadId)
{
  /*Create variables to store intermediate results. Circumvent false sharing*/
  unsigned long    numberOfPixelsCounted = 0;
  MeasureType      measure = NumericTraits<MeasureType>::Zero;
  DerivativeType & derivative = this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Derivative;

  /** Array that stores dM(x)/dmu, and the sparse jacobian+indices. */
  NonZeroJacobianIndicesType nzji(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  DerivativeType             imageJacobian(nzji.size());
  TransformJacobianType      jacobian;

  /** Matrix to store the spatial Jacobian, dT/dx. */
  SpatialJacobianType spatialJac;

  /** Matrix to store the scaled inverse spatial Jacobian, det(dT/dx) * (dT/dx)^-1 */
  SpatialJacobianType inverseSpatialJacobian;

  /** Array that stores JacobianOfSpatialJacobian, d(dT/dx)/dmu */
  JacobianOfSpatialJacobianType jacobianOfSpatialJacobian;

  DerivativeType jacobianOfSpatialJacobianDeterminant(nzji.size());

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nSamplesPerThread = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  unsigned long pos_begin = nSamplesPerThread * threadId;
  unsigned long pos_end = nSamplesPerThread * (threadId + 1);
  pos_begin = (pos_begin > sampleContainerSize) ? sampleContainerSize : pos_begin;
  pos_end = (pos_end > sampleContainerSize) ? sampleContainerSize : pos_end;

  /** Create iterator over the sample container. */
  typename ImageSampleContainerType::ConstIterator threader_fiter;
  typename ImageSampleContainerType::ConstIterator threader_fbegin = sampleContainer->Begin();
  typename ImageSampleContainerType::ConstIterator threader_fend = sampleContainer->Begin();
  threader_fbegin += (int)pos_begin;
  threader_fend += (int)pos_end;

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

      /** Get the TransformJacobian dT/dmu. */
      this->EvaluateTransformJacobian(fixedPoint, jacobian, nzji);

      /** Compute the inner products (dM/dx)^T (dT/dmu). */
      this->EvaluateTransformJacobianInnerProduct(jacobian, movingImageDerivative, imageJacobian);

      /** Get the SpatialJacobian dT/dx. */
      this->m_AdvancedTransform->GetSpatialJacobian(fixedPoint, spatialJac);

      /** Compute the determinant of the Transform Jacobian |dT/dx|. */
      const RealType detjac = static_cast<RealType>(vnl_det(spatialJac.GetVnlMatrix()));

      /** Compute the inverse spatialJacobian. */
      inverseSpatialJacobian = spatialJac.GetInverse();

      /** Compute the JacobianOfSpatialJacobian. */
      this->m_AdvancedTransform->GetJacobianOfSpatialJacobian(fixedPoint, jacobianOfSpatialJacobian, nzji);

      /** Compute the dot product of the inverse spatialJacobian and JacobianOfSpatialJacobian
       * to support calculation of the JacobianOfSpatialJacobianDeterminant.
       */
      this->EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct(
        jacobianOfSpatialJacobian, inverseSpatialJacobian, jacobianOfSpatialJacobianDeterminant);

      /** Compute this pixel's contribution to the measure and derivatives. */
      this->UpdateValueAndDerivativeTerms(fixedImageValue,
                                          movingImageValue,
                                          imageJacobian,
                                          nzji,
                                          detjac,
                                          jacobianOfSpatialJacobianDeterminant,
                                          measure,
                                          derivative);

    } // end if sampleOk
  }

  /** Only update these variables at the end to prevent unnecessary "false sharing". */
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_NumberOfPixelsCounted = numberOfPixelsCounted;
  this->m_GetValueAndDerivativePerThreadVariables[threadId].st_Value = measure;
} // end ThreadedGetValueAndDerivative()


/**
 * *************** AfterThreadedGetValueAndDerivative ****************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
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

  /** Accumulate values. */
  value = NumericTraits<MeasureType>::Zero;
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    value += this->m_GetValueAndDerivativePerThreadVariables[i].st_Value;

    /** Reset this variable for the next iteration. */
    this->m_GetValueAndDerivativePerThreadVariables[i].st_Value = NumericTraits<MeasureType>::Zero;
  }

  value /= static_cast<RealType>(this->m_NumberOfPixelsCounted);

  /** Accumulate derivatives. */
  /** compute single-threadedly */
  if (!this->m_UseMultiThread && false) // force multi-threaded as in AdvancedMeanSquares
  {
    derivative = this->m_GetValueAndDerivativePerThreadVariables[0].st_Derivative;
    for (ThreadIdType i = 1; i < numberOfThreads; ++i)
    {
      derivative += this->m_GetValueAndDerivativePerThreadVariables[i].st_Derivative;
    }

    derivative /= static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);
  }
  // compute multi-threadedly with itk threads
  else if (true) // force ITK threads !this->m_UseOpenMP )
  {
    this->m_ThreaderMetricParameters.st_DerivativePointer = derivative.begin();
    this->m_ThreaderMetricParameters.st_NormalizationFactor =
      static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);

    this->m_Threader->SetSingleMethod(this->AccumulateDerivativesThreaderCallback,
                                      const_cast<void *>(static_cast<const void *>(&this->m_ThreaderMetricParameters)));
    this->m_Threader->SingleMethodExecute();
  }

#ifdef ELASTIX_USE_OPENMP
  /** compute multi-threadedly with openmp.  Never used? */
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
      derivative[j] = tmp / static_cast<DerivativeValueType>(this->m_NumberOfPixelsCounted);
    }
  }
#endif

} // end AfterThreadedGetValueAndDerivative()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  using JacobianIteratorType = typename TransformJacobianType::const_iterator;
  JacobianIteratorType jac = jacobian.begin();
  imageJacobian.Fill(0.0);

  for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
  {
    const double imDeriv = movingImageDerivative[dim] / (this->m_TissueValue - this->m_AirValue);

    for (auto & imageJacobianElement : imageJacobian)
    {
      imageJacobianElement += (*jac) * imDeriv;
      ++jac;
    }
  }

} // end EvaluateTransformJacobianInnerProduct()


/**
 * *************** UpdateValueAndDerivativeTerms ***************************
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::UpdateValueAndDerivativeTerms(
  const RealType                     fixedImageValue,
  const RealType                     movingImageValue,
  const DerivativeType &             imageJacobian,
  const NonZeroJacobianIndicesType & nzji,
  const RealType                     spatialJacobianDeterminant,
  const DerivativeType &             jacobianOfSpatialJacobianDeterminant,
  MeasureType &                      measure,
  DerivativeType &                   deriv) const
{
  /** The difference squared. */
  const RealType diff =
    ((fixedImageValue - this->m_AirValue) - spatialJacobianDeterminant * (movingImageValue - this->m_AirValue)) /
    (this->m_TissueValue - this->m_AirValue);
  const RealType diffdiff = diff * diff;
  measure += diffdiff;

  /** Calculate the contributions to the derivatives with respect to each parameter. */
  const RealType diff_2 = diff * -2.0;

  const auto numberOfParameters = this->GetNumberOfParameters();

  if (nzji.size() == numberOfParameters)
  {
    /** Loop over all Jacobians. */
    typename DerivativeType::const_iterator imjacit = imageJacobian.begin();
    typename DerivativeType::const_iterator jsjdit = jacobianOfSpatialJacobianDeterminant.begin();
    typename DerivativeType::iterator       derivit = deriv.begin();
    for (unsigned int mu = 0; mu < numberOfParameters; ++mu)
    {
      (*derivit) +=
        diff_2 * spatialJacobianDeterminant *
        ((*jsjdit) * (movingImageValue - this->m_AirValue) / (this->m_TissueValue - this->m_AirValue) + (*imjacit));
      ++imjacit;
      ++jsjdit;
      ++derivit;
    }
  }
  else
  {
    /** Only pick the nonzero Jacobians. */
    for (unsigned int i = 0; i < imageJacobian.GetSize(); ++i)
    {
      const unsigned int index = nzji[i];
      deriv[index] += diff_2 * spatialJacobianDeterminant *
                      (jacobianOfSpatialJacobianDeterminant[i] * (movingImageValue - this->m_AirValue) /
                         (this->m_TissueValue - this->m_AirValue) +
                       imageJacobian[i]);
    }
  }
} // end UpdateValueAndDerivativeTerms()


/**
 * *************** EvaluateInverseSpatialJacobian **************************
 */

template <class TFixedImage, class TMovingImage>
bool
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::EvaluateInverseSpatialJacobian(
  const SpatialJacobianType & spatialJacobian,
  const RealType              spatialJacobianDeterminant,
  SpatialJacobianType &       inverseSpatialJacobian) const
{
  inverseSpatialJacobian.Fill(0.0);
  inverseSpatialJacobian = spatialJacobian.GetInverse();

  return true;

} // end EvaluateInverseSpatialJacobian()


/**
 * ********** EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct ******
 */

template <class TFixedImage, class TMovingImage>
void
SumSquaredTissueVolumeDifferenceImageToImageMetric<TFixedImage, TMovingImage>::
  EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct(
    const JacobianOfSpatialJacobianType & jacobianOfSpatialJacobian,
    const SpatialJacobianType &           inverseSpatialJacobian,
    DerivativeType &                      jacobianOfSpatialJacobianDeterminant) const
{
  using JacobianOfSpatialJacobianIteratorType = typename JacobianOfSpatialJacobianType::const_iterator;
  using DerivativeIteratorType = typename DerivativeType::iterator;

  jacobianOfSpatialJacobianDeterminant.Fill(0.0);

  JacobianOfSpatialJacobianIteratorType jsjit = jacobianOfSpatialJacobian.begin();
  DerivativeIteratorType                jsjdit = jacobianOfSpatialJacobianDeterminant.begin();

  const unsigned int sizejacobianOfSpatialJacobianDeterminant = jacobianOfSpatialJacobianDeterminant.GetSize();

  /** matrix product first, then trace. */
  for (unsigned int mu = 0; mu < sizejacobianOfSpatialJacobianDeterminant; ++mu)
  {
    for (unsigned int diag = 0; diag < FixedImageDimension; ++diag)
    {
      for (unsigned int idx = 0; idx < FixedImageDimension; ++idx)
      {
        (*jsjdit) += inverseSpatialJacobian(diag, idx) * (*jsjit)(idx, diag);
      }
    }
    ++jsjdit;
    ++jsjit;
  }

} // end EvaluateJacobianOfSpatialJacobianDeterminantInnerProduct()


} // end namespace itk

#endif // end #ifndef _itkSumSquaredTissueVolumeDifferenceImageToImageMetric_hxx
