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
#ifndef _itkAdvancedImageToImageMetric_hxx
#define _itkAdvancedImageToImageMetric_hxx

#include "itkAdvancedImageToImageMetric.h"
#include "elxDefaultConstruct.h"

#include "itkAdvancedRayCastInterpolateImageFunction.h"
#include "itkComputeImageExtremaFilter.h"
#include <itkDeref.h>

#include <algorithm> // For min.
#include <cassert>

namespace itk
{

/**
 * ********************* Constructor ****************************
 */

template <typename TFixedImage, typename TMovingImage>
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::AdvancedImageToImageMetric()
{
  /** Don't use the default gradient image as implemented by ITK.
   * It uses a Gaussian derivative, which introduces extra smoothing,
   * which may not always be desired. Also, when the derivatives are
   * computed using Gaussian filtering, the gray-values should also be
   * blurred, to have a consistent 'image model'.
   */
  this->SetComputeGradient(false);

  /** Initialize the m_ThreaderMetricParameters. */
  m_ThreaderMetricParameters.st_Metric = this;

} // end Constructor


/**
 * ********************* Initialize ****************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();

  /** Setup the parameters for the gray value limiters. */
  this->InitializeLimiters();

  /** Connect the image sampler */
  this->InitializeImageSampler();

  /** Check if the interpolator is a B-spline interpolator. */
  this->CheckForBSplineInterpolator();

  /** Check if the transform is an advanced transform. */
  this->CheckForAdvancedTransform();

  /** Check if the transform is a B-spline transform. */
  this->CheckForBSplineTransform();

  /** Initialize some threading related parameters. */
  if (m_UseMultiThread)
  {
    this->InitializeThreadingParameters();

    const auto setNumberOfWorkUnitsIfNotNull = [this](const auto bsplineInterpolator) {
      if (!bsplineInterpolator.IsNull())
      {
        bsplineInterpolator->SetNumberOfWorkUnits(this->Superclass::GetNumberOfWorkUnits());
      }
    };
    setNumberOfWorkUnitsIfNotNull(m_BSplineInterpolator);
    setNumberOfWorkUnitsIfNotNull(m_BSplineInterpolatorFloat);
  }

} // end Initialize()


/**
 * ********************* InitializeThreadingParameters ****************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  /** Resize and initialize the threading related parameters.
   * The SetSize() functions do not resize the data when this is not
   * needed, which saves valuable re-allocation time.
   *
   * This function is only to be called at the start of each resolution.
   * Re-initialization of the potentially large vectors is performed after
   * each iteration, in the accumulate functions, in a multi-threaded fashion.
   * This has performance benefits for larger vector sizes.
   */

  /** Only resize the array of structs when needed. */
  if (m_GetValueAndDerivativePerThreadVariablesSize != numberOfThreads)
  {
    m_GetValueAndDerivativePerThreadVariables.reset(new AlignedGetValueAndDerivativePerThreadStruct[numberOfThreads]);
    m_GetValueAndDerivativePerThreadVariablesSize = numberOfThreads;
  }

  /** Some initialization. */
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    m_GetValueAndDerivativePerThreadVariables[i].st_NumberOfPixelsCounted = SizeValueType{};
    m_GetValueAndDerivativePerThreadVariables[i].st_Value = MeasureType{};
    m_GetValueAndDerivativePerThreadVariables[i].st_Derivative.SetSize(this->GetNumberOfParameters());
    m_GetValueAndDerivativePerThreadVariables[i].st_Derivative.Fill(0.0);
  }

} // end InitializeThreadingParameters()


/**
 * ****************** InitializeLimiters *****************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeLimiters()
{
  /** Set up fixed limiter. */
  if (m_UseFixedImageLimiter)
  {
    if (m_FixedImageLimiter == nullptr)
    {
      itkExceptionMacro("No fixed image limiter has been set!");
    }

    const auto computeFixedImageExtrema = ComputeImageExtremaFilter<FixedImageType>::New();
    computeFixedImageExtrema->SetInput(this->GetFixedImage());
    computeFixedImageExtrema->SetImageSpatialMask(this->GetFixedImageMask());
    computeFixedImageExtrema->Update();

    m_FixedImageTrueMax = computeFixedImageExtrema->GetMaximum();
    m_FixedImageTrueMin = computeFixedImageExtrema->GetMinimum();

    m_FixedImageMinLimit = static_cast<FixedImageLimiterOutputType>(
      m_FixedImageTrueMin - m_FixedLimitRangeRatio * (m_FixedImageTrueMax - m_FixedImageTrueMin));
    m_FixedImageMaxLimit = static_cast<FixedImageLimiterOutputType>(
      m_FixedImageTrueMax + m_FixedLimitRangeRatio * (m_FixedImageTrueMax - m_FixedImageTrueMin));

    m_FixedImageLimiter->SetLowerThreshold(static_cast<RealType>(m_FixedImageTrueMin));
    m_FixedImageLimiter->SetUpperThreshold(static_cast<RealType>(m_FixedImageTrueMax));
    m_FixedImageLimiter->SetLowerBound(m_FixedImageMinLimit);
    m_FixedImageLimiter->SetUpperBound(m_FixedImageMaxLimit);

    m_FixedImageLimiter->Initialize();
  }

  /** Set up moving limiter. */
  if (m_UseMovingImageLimiter)
  {
    if (m_MovingImageLimiter == nullptr)
    {
      itkExceptionMacro("No moving image limiter has been set!");
    }

    const auto computeMovingImageExtrema = ComputeImageExtremaFilter<MovingImageType>::New();
    computeMovingImageExtrema->SetInput(this->GetMovingImage());
    computeMovingImageExtrema->SetImageSpatialMask(this->GetMovingImageMask());
    computeMovingImageExtrema->Update();

    m_MovingImageTrueMax = computeMovingImageExtrema->GetMaximum();
    m_MovingImageTrueMin = computeMovingImageExtrema->GetMinimum();

    m_MovingImageMinLimit = static_cast<MovingImageLimiterOutputType>(
      m_MovingImageTrueMin - m_MovingLimitRangeRatio * (m_MovingImageTrueMax - m_MovingImageTrueMin));
    m_MovingImageMaxLimit = static_cast<MovingImageLimiterOutputType>(
      m_MovingImageTrueMax + m_MovingLimitRangeRatio * (m_MovingImageTrueMax - m_MovingImageTrueMin));

    m_MovingImageLimiter->SetLowerThreshold(static_cast<RealType>(m_MovingImageTrueMin));
    m_MovingImageLimiter->SetUpperThreshold(static_cast<RealType>(m_MovingImageTrueMax));
    m_MovingImageLimiter->SetLowerBound(m_MovingImageMinLimit);
    m_MovingImageLimiter->SetUpperBound(m_MovingImageMaxLimit);

    m_MovingImageLimiter->Initialize();
  }

} // end InitializeLimiters()


/**
 * ********************* InitializeImageSampler ****************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::InitializeImageSampler()
{
  if (m_UseImageSampler)
  {
    /** Check if the ImageSampler is set. */
    if (!m_ImageSampler)
    {
      itkExceptionMacro("ImageSampler is not present");
    }

    /** Initialize the Image Sampler. */
    m_ImageSampler->SetInput(Superclass::m_FixedImage);
    m_ImageSampler->SetMask(this->GetFixedImageMask());
    m_ImageSampler->SetInputImageRegion(this->GetFixedImageRegion());
  }

} // end InitializeImageSampler()


/**
 * ****************** CheckForBSplineInterpolator **********************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForBSplineInterpolator()
{
  /** Check if the interpolator is of type BSplineInterpolateImageFunction,
   * or of type AdvancedLinearInterpolateImageFunction.
   * If so, we can make use of their EvaluateDerivatives methods.
   * Otherwise, we precompute the gradients using a central difference scheme,
   * and do evaluate the gradient using nearest neighbor interpolation.
   */
  InterpolatorType * const interpolator = Superclass::m_Interpolator.GetPointer();

  m_BSplineInterpolator = dynamic_cast<BSplineInterpolatorType *>(interpolator);
  if (m_BSplineInterpolator)
  {
    itkDebugMacro("Interpolator is B-spline");
  }
  else
  {
    itkDebugMacro("Interpolator is not B-spline");
  }

  m_BSplineInterpolatorFloat = dynamic_cast<BSplineInterpolatorFloatType *>(interpolator);
  if (m_BSplineInterpolatorFloat)
  {
    itkDebugMacro("Interpolator is BSplineFloat");
  }
  else
  {
    itkDebugMacro("Interpolator is not BSplineFloat");
  }

  m_ReducedBSplineInterpolator = dynamic_cast<ReducedBSplineInterpolatorType *>(interpolator);
  if (m_ReducedBSplineInterpolator)
  {
    itkDebugMacro("Interpolator is ReducedBSpline");
  }
  else
  {
    itkDebugMacro("Interpolator is not ReducedBSpline");
  }

  m_LinearInterpolator = dynamic_cast<LinearInterpolatorType *>(interpolator);

  /** Don't overwrite the gradient image if m_ComputeGradient == true.
   * Otherwise we can use a forward difference derivative, or the derivative
   * provided by the B-spline interpolator.
   */
  if (!Superclass::m_ComputeGradient)
  {
    /** In addition, don't compute the moving image gradient for 2D/3D registration,
     * i.e. whenever the interpolator is a ray cast interpolator.
     * This is a bit of a hack that does not respect the setting of the boolean
     * m_ComputeGradient. By doing this, there is no way to ask no gradient
     * computation at all (to save memory).
     * The best solution would be to remove everything below this point, and to
     * override the ComputeGradient() function of ITK by computing a central
     * difference derivative. This way SetComputeGradient will enable or disable
     * the gradient computation and let derived classes choose if it needs the
     * precomputation of the gradient.
     *
     * For more details see the post about "2D/3D registration memory issue" in
     * elastix's mailing list (2 July 2012).
     */
    using NearestNeighborInterpolatorType =
      NearestNeighborInterpolateImageFunction<MovingImageType, CoordinateRepresentationType>;

    if (dynamic_cast<NearestNeighborInterpolatorType *>(interpolator))
    {
      using CentralDifferenceGradientFilterType = GradientImageFilter<TMovingImage, RealType, RealType>;

      elx::DefaultConstruct<CentralDifferenceGradientFilterType> centralDifferenceGradientFilter{};
      centralDifferenceGradientFilter.SetUseImageSpacing(true);
      centralDifferenceGradientFilter.SetInput(Superclass::m_MovingImage);
      centralDifferenceGradientFilter.Update();
      Superclass::m_GradientImage = centralDifferenceGradientFilter.GetOutput();
    }
    else
    {
      using RayCastInterpolatorType =
        itk::AdvancedRayCastInterpolateImageFunction<MovingImageType, CoordinateRepresentationType>;

      assert(m_BSplineInterpolator || m_BSplineInterpolatorFloat || m_ReducedBSplineInterpolator ||
             m_LinearInterpolator || dynamic_cast<RayCastInterpolatorType *>(interpolator));

      Superclass::m_GradientImage = nullptr;
    }
  }

} // end CheckForBSplineInterpolator()


/**
 * ****************** CheckForAdvancedTransform **********************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForAdvancedTransform()
{
  /** Check if the transform is of type AdvancedTransform. */
  m_AdvancedTransform = dynamic_cast<AdvancedTransformType *>(Superclass::m_Transform.GetPointer());
  if (!m_AdvancedTransform)
  {
    itkDebugMacro("Transform is not Advanced");
    itkExceptionMacro("The AdvancedImageToImageMetric requires an AdvancedTransform");
  }
  else
  {
    itkDebugMacro("Transform is Advanced");
  }

} // end CheckForAdvancedTransform()


/**
 * ****************** CheckForBSplineTransform **********************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckForBSplineTransform() const
{
  /** Check if this transform is a combo transform. */
  auto * testPtr_combo = dynamic_cast<CombinationTransformType *>(m_AdvancedTransform.GetPointer());

  /** Check if this transform is a B-spline transform. */
  auto * testPtr_1 = dynamic_cast<BSplineOrder1TransformType *>(m_AdvancedTransform.GetPointer());
  auto * testPtr_2 = dynamic_cast<BSplineOrder2TransformType *>(m_AdvancedTransform.GetPointer());
  auto * testPtr_3 = dynamic_cast<BSplineOrder3TransformType *>(m_AdvancedTransform.GetPointer());

  bool transformIsBSpline = false;
  if (testPtr_1 || testPtr_2 || testPtr_3)
  {
    transformIsBSpline = true;
  }
  else if (testPtr_combo)
  {
    /** Check if the current transform is a B-spline transform. */
    const auto * testPtr_1b = dynamic_cast<const BSplineOrder1TransformType *>(testPtr_combo->GetCurrentTransform());
    const auto * testPtr_2b = dynamic_cast<const BSplineOrder2TransformType *>(testPtr_combo->GetCurrentTransform());
    const auto * testPtr_3b = dynamic_cast<const BSplineOrder3TransformType *>(testPtr_combo->GetCurrentTransform());
    if (testPtr_1b || testPtr_2b || testPtr_3b)
    {
      transformIsBSpline = true;
    }
  }

  /** Store the result. */
  m_TransformIsBSpline = transformIsBSpline;

} // end CheckForBSplineTransform()


/**
 * ******************* EvaluateMovingImageValueAndDerivativeWithOptionalThreadId ******************
 */

template <typename TFixedImage, typename TMovingImage>
template <typename... TOptionalThreadId>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateMovingImageValueAndDerivativeWithOptionalThreadId(
  const MovingImagePointType & mappedPoint,
  RealType &                   movingImageValue,
  MovingImageDerivativeType *  gradient,
  const TOptionalThreadId... optionalThreadId) const
{
  /** Check if mapped point inside image buffer. */
  MovingImageContinuousIndexType cindex;
  Superclass::m_Interpolator->ConvertPointToContinuousIndex(mappedPoint, cindex);
  bool sampleOk = Superclass::m_Interpolator->IsInsideBuffer(cindex);
  if (sampleOk)
  {
    /** Compute value and possibly derivative. */
    if (gradient)
    {
      if (m_BSplineInterpolator && !Superclass::m_ComputeGradient)
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        m_BSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient, optionalThreadId...);
      }
      else if (m_BSplineInterpolatorFloat && !Superclass::m_ComputeGradient)
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        m_BSplineInterpolatorFloat->EvaluateValueAndDerivativeAtContinuousIndex(
          cindex, movingImageValue, *gradient, optionalThreadId...);
      }
      else if (m_ReducedBSplineInterpolator && !Superclass::m_ComputeGradient)
      {
        /** Compute moving image value and gradient using the B-spline kernel. */
        movingImageValue = Superclass::m_Interpolator->EvaluateAtContinuousIndex(cindex);
        (*gradient) = m_ReducedBSplineInterpolator->EvaluateDerivativeAtContinuousIndex(cindex);
        // m_ReducedBSplineInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(
        //  cindex, movingImageValue, *gradient );
      }
      else if (m_LinearInterpolator && !Superclass::m_ComputeGradient)
      {
        /** Compute moving image value and gradient using the linear interpolator. */
        m_LinearInterpolator->EvaluateValueAndDerivativeAtContinuousIndex(cindex, movingImageValue, *gradient);
      }
      else
      {
        /** Get the gradient by NearestNeighboorInterpolation of the gradient image.
         * It is assumed that the gradient image is computed.
         */
        movingImageValue = Superclass::m_Interpolator->EvaluateAtContinuousIndex(cindex);
        MovingImageIndexType index;
        for (unsigned int j = 0; j < MovingImageDimension; ++j)
        {
          index[j] = Math::Round<int64_t>(cindex[j]);
        }
        (*gradient) = Superclass::m_GradientImage->GetPixel(index);
      }

      /** The moving image gradient is multiplied with its scales, when requested. */
      if (m_UseMovingImageDerivativeScales)
      {
        if (!m_ScaleGradientWithRespectToMovingImageOrientation)
        {
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            (*gradient)[i] *= m_MovingImageDerivativeScales[i];
          }
        }
        else
        {
          /** Optionally, the scales are applied with respect to the moving image orientation.
           * The above default option implicitly applies the scales with respect to the
           * orientation of the transformation axis. In some cases you may want to restrict
           * moving image motion with respect to its own axes. This is achieved below by pre
           * and post rotation by the direction cosines of the moving image.
           * First the gradient is rotated backwards to a standardized axis.
           */
          using InternalMatrixType = typename MovingImageType::DirectionType::InternalMatrixType;
          const InternalMatrixType M = this->GetMovingImage()->GetDirection().GetVnlMatrix();
          vnl_vector<double>       rotated_gradient_vnl = M.transpose() * gradient->GetVnlVector();

          /** Then scales are applied. */
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            rotated_gradient_vnl[i] *= m_MovingImageDerivativeScales[i];
          }

          /** The scaled gradient is then rotated forwards again. */
          rotated_gradient_vnl = M * rotated_gradient_vnl;

          /** Copy the vnl version back to the original. */
          for (unsigned int i = 0; i < MovingImageDimension; ++i)
          {
            (*gradient)[i] = rotated_gradient_vnl[i];
          }
        }
      } // end if m_UseMovingImageDerivativeScales
    } // end if gradient
    else
    {
      movingImageValue = Superclass::m_Interpolator->EvaluateAtContinuousIndex(cindex);
    }
  } // end if sampleOk

  return sampleOk;

} // end EvaluateMovingImageValueAndDerivativeWithOptionalThreadId()


/**
 * *************** EvaluateTransformJacobianInnerProduct ****************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobianInnerProduct(
  const TransformJacobianType &     jacobian,
  const MovingImageDerivativeType & movingImageDerivative,
  DerivativeType &                  imageJacobian) const
{
  /** Multiple the 1-by-dim vector movingImageDerivative with the
   * dim-by-length matrix jacobian, to get a 1-by-length vector imageJacobian.
   * An optimized route can be taken for B-spline transforms.
   */
  if (m_TransformIsBSpline)
  {
    // For the B-spline we know that the Jacobian is mostly empty.
    //       [ j ... j 0 ... 0 0 ... 0 ]
    // jac = [ 0 ... 0 j ... j 0 ... 0 ]
    //       [ 0 ... 0 0 ... 0 j ... j ]
    const unsigned int sizeImageJacobian = imageJacobian.GetSize();
    const unsigned int numberOfParametersPerDimension = sizeImageJacobian / FixedImageDimension;
    unsigned int       counter = 0;
    for (unsigned int dim = 0; dim < FixedImageDimension; ++dim)
    {
      const double imDeriv = movingImageDerivative[dim];
      for (unsigned int mu = 0; mu < numberOfParametersPerDimension; ++mu)
      {
        imageJacobian(counter) = jacobian(dim, counter) * imDeriv;
        ++counter;
      }
    }
  }
  else
  {
    /** Otherwise perform a full multiplication. */
    ImplementationDetails::EvaluateInnerProduct(jacobian, movingImageDerivative, imageJacobian);
  }
} // end EvaluateTransformJacobianInnerProduct()


/**
 * ********************** TransformPoint ************************
 */

template <typename TFixedImage, typename TMovingImage>
auto
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::TransformPoint(const FixedImagePointType & fixedImagePoint) const
  -> MovingImagePointType
{
  return Superclass::m_Transform->TransformPoint(fixedImagePoint);

} // end TransformPoint()


/**
 * *************** EvaluateTransformJacobian ****************
 */

template <typename TFixedImage, typename TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::EvaluateTransformJacobian(
  const FixedImagePointType &  fixedImagePoint,
  TransformJacobianType &      jacobian,
  NonZeroJacobianIndicesType & nzji) const
{
  /** Advanced transform: generic sparse Jacobian support */
  m_AdvancedTransform->GetJacobian(fixedImagePoint, jacobian, nzji);

  /** For future use: return whether the sample is valid */
  const bool valid = true;
  return valid;

} // end EvaluateTransformJacobian()


/**
 * ************************** IsInsideMovingMask *************************
 */

template <typename TFixedImage, typename TMovingImage>
bool
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::IsInsideMovingMask(const MovingImagePointType & point) const
{
  /** If a mask has been set: */
  if (const auto * const mask = this->GetMovingImageMask())
  {
    return mask->IsInsideInWorldSpace(point);
  }

  /** If no mask has been set, just return true. */
  return true;

} // end IsInsideMovingMask()


/**
 * *********************** BeforeThreadedGetValueAndDerivative ***********************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::BeforeThreadedGetValueAndDerivative(
  const TransformParametersType & parameters) const
{
  /** In this function do all stuff that cannot be multi-threaded. */
  if (m_UseMetricSingleThreaded)
  {
    this->SetTransformParameters(parameters);
    if (m_UseImageSampler)
    {
      m_ImageSampler->Update();
    }
  }

} // end BeforeThreadedGetValueAndDerivative()


/**
 * **************** GetValueThreaderCallback *******
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_TYPE
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValueThreaderCallback(void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadID = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

  assert(userData.st_Metric);
  const Self & metric = *(userData.st_Metric);

  metric.ThreadedGetValue(threadID);

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end GetValueThreaderCallback()


/**
 * *********************** LaunchGetValueThreaderCallback***************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::LaunchGetValueThreaderCallback() const
{
  /** Setup threader and launch. */
  Superclass::m_Threader->SetSingleMethodAndExecute(this->GetValueThreaderCallback, &m_ThreaderMetricParameters);

} // end LaunchGetValueThreaderCallback()


/**
 * **************** GetValueAndDerivativeThreaderCallback *******
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeThreaderCallback(void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadID = infoStruct.WorkUnitID;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

  assert(userData.st_Metric);
  const Self & metric = *(userData.st_Metric);

  metric.ThreadedGetValueAndDerivative(threadID);

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end GetValueAndDerivativeThreaderCallback()


/**
 * *********************** LaunchGetValueAndDerivativeThreaderCallback***************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::LaunchGetValueAndDerivativeThreaderCallback() const
{
  /** Setup threader and launch. */
  Superclass::m_Threader->SetSingleMethodAndExecute(this->GetValueAndDerivativeThreaderCallback,
                                                    &m_ThreaderMetricParameters);

} // end LaunchGetValueAndDerivativeThreaderCallback()


/**
 *********** AccumulateDerivativesThreaderCallback *************
 */

template <typename TFixedImage, typename TMovingImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::AccumulateDerivativesThreaderCallback(void * arg)
{
  assert(arg);
  const auto & infoStruct = *static_cast<ThreadInfoType *>(arg);
  ThreadIdType threadID = infoStruct.WorkUnitID;
  ThreadIdType nrOfThreads = infoStruct.NumberOfWorkUnits;

  assert(infoStruct.UserData);
  const auto & userData = *static_cast<MultiThreaderParameterType *>(infoStruct.UserData);

  assert(userData.st_Metric);
  Self & metric = *(userData.st_Metric);

  const unsigned int numPar = metric.GetNumberOfParameters();
  const auto         subSize =
    static_cast<unsigned int>(std::ceil(static_cast<double>(numPar) / static_cast<double>(nrOfThreads)));
  const unsigned int jmin = threadID * subSize;
  const unsigned int jmax = std::min((threadID + 1) * subSize, numPar);

  /** This thread accumulates all sub-derivatives into a single one, for the
   * range [ jmin, jmax [. Additionally, the sub-derivatives are reset.
   */
  const DerivativeValueType normalization = 1.0 / userData.st_NormalizationFactor;
  for (unsigned int j = jmin; j < jmax; ++j)
  {
    DerivativeValueType sum{};
    for (ThreadIdType i = 0; i < nrOfThreads; ++i)
    {
      sum += metric.m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j];

      /** Reset this variable for the next iteration. */
      metric.m_GetValueAndDerivativePerThreadVariables[i].st_Derivative[j] = 0.0;
    }
    userData.st_DerivativePointer[j] = sum * normalization;
  }

  return ITK_THREAD_RETURN_DEFAULT_VALUE;

} // end AccumulateDerivativesThreaderCallback()


/**
 * *********************** CheckNumberOfSamples ***********************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::CheckNumberOfSamples() const
{
  const auto & samples = Deref(Deref(m_ImageSampler.get()).GetOutput());
  const auto   numberOfSamples = samples.size();

  if (const SizeValueType found{ Superclass::m_NumberOfPixelsCounted };
      found < m_RequiredRatioOfValidSamples * numberOfSamples)
  {
    itkExceptionMacro("Too many samples map outside moving image buffer: " << found << " / " << numberOfSamples
                                                                           << '\n');
  }

} // end CheckNumberOfSamples()


/**
 * ********************* PrintSelf ****************************
 */

template <typename TFixedImage, typename TMovingImage>
void
AdvancedImageToImageMetric<TFixedImage, TMovingImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  /** Variables related to the Sampler. */
  os << indent << "Variables related to the Sampler: " << std::endl;
  os << indent.GetNextIndent() << "ImageSampler: " << m_ImageSampler.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "UseImageSampler: " << m_UseImageSampler << std::endl;

  /** Variables for the Limiters. */
  os << indent << "Variables related to the Limiters: " << std::endl;
  os << indent.GetNextIndent() << "FixedLimitRangeRatio: " << m_FixedLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "MovingLimitRangeRatio: " << m_MovingLimitRangeRatio << std::endl;
  os << indent.GetNextIndent() << "UseFixedImageLimiter: " << m_UseFixedImageLimiter << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageLimiter: " << m_UseMovingImageLimiter << std::endl;
  os << indent.GetNextIndent() << "FixedImageLimiter: " << m_FixedImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "MovingImageLimiter: " << m_MovingImageLimiter.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMin: " << m_FixedImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMin: " << m_MovingImageTrueMin << std::endl;
  os << indent.GetNextIndent() << "FixedImageTrueMax: " << m_FixedImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "MovingImageTrueMax: " << m_MovingImageTrueMax << std::endl;
  os << indent.GetNextIndent() << "FixedImageMinLimit: " << m_FixedImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMinLimit: " << m_MovingImageMinLimit << std::endl;
  os << indent.GetNextIndent() << "FixedImageMaxLimit: " << m_FixedImageMaxLimit << std::endl;
  os << indent.GetNextIndent() << "MovingImageMaxLimit: " << m_MovingImageMaxLimit << std::endl;

  /** Variables related to image derivative computation. */
  os << indent << "Variables related to image derivative computation: " << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolator: " << m_BSplineInterpolator.GetPointer() << std::endl;
  os << indent.GetNextIndent() << "BSplineInterpolatorFloat: " << m_BSplineInterpolatorFloat.GetPointer() << std::endl;

  /** Variables used when the transform is a B-spline transform. */
  os << indent << "Variables store the transform as an AdvancedTransform: " << std::endl;
  os << indent.GetNextIndent() << "AdvancedTransform: " << m_AdvancedTransform.GetPointer() << std::endl;

  /** Other variables. */
  os << indent << "Other variables of the AdvancedImageToImageMetric: " << std::endl;
  os << indent.GetNextIndent() << "RequiredRatioOfValidSamples: " << m_RequiredRatioOfValidSamples << std::endl;
  os << indent.GetNextIndent() << "UseMovingImageDerivativeScales: " << m_UseMovingImageDerivativeScales << std::endl;
  os << indent.GetNextIndent() << "MovingImageDerivativeScales: " << m_MovingImageDerivativeScales << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef _itkAdvancedImageToImageMetric_hxx
