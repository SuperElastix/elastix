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
#ifndef itkAdvancedImageToImageMetric_h
#define itkAdvancedImageToImageMetric_h

#include "itkImageToImageMetric.h"

#include "itkImageSamplerBase.h"
#include "itkGradientImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkReducedDimensionBSplineInterpolateImageFunction.h"
#include "itkAdvancedLinearInterpolateImageFunction.h"
#include "itkLimiterFunctionBase.h"
#include "itkFixedArray.h"
#include "itkAdvancedTransform.h"
#include <vnl/vnl_sparse_matrix.h>

#include "itkImageMaskSpatialObject.h"

// Needed for checking for B-spline for faster implementation
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

#include "itkPlatformMultiThreader.h"

#include <memory> // For unique_ptr.

namespace itk
{

/** \class AdvancedImageToImageMetric
 *
 * \brief An extension of the ITK ImageToImageMetric. It is the intended base
 * class for all elastix metrics.
 *
 * This class inherits from the itk::ImageToImageMetric. The additional features of
 * this class that makes it an AdvancedImageToImageMetric are:
 * \li The use of an ImageSampler, which selects the fixed image samples over which
 *   the metric is evaluated. In the derived metric you simply need to loop over
 *   the image sample container, instead over the fixed image. This way it is easy
 *   to create different samplers, without the derived metric needing to know.
 * \li Gray value limiters: for some metrics it is important to know the range of expected
 *   gray values in the fixed and moving image, beforehand. However, when a third order
 *   B-spline interpolator is used to interpolate the images, the interpolated values may
 *   be larger than the range of voxel values, because of so-called overshoot. The
 *   gray-value limiters make sure this doesn't happen.
 * \li Fast implementation when a B-spline transform is used. The B-spline transform
 *   has a sparse Jacobian. The AdvancedImageToImageMetric provides functions that make
 *   it easier for inheriting metrics to exploit this fact.
 * \li MovingImageDerivativeScales: an experimental option, which allows scaling of the
 *   moving image derivatives. This is a kind of fast hack, which makes it possible to
 *   avoid transformation in one direction (x, y, or z). Do not use this functionality
 *   unless you have a good reason for it...
 * \li Some convenience functions are provided, such as the IsInsideMovingMask
 *   and CheckNumberOfSamples.
 *
 * The parameters used in this class are:
 * \parameter MovingImageDerivativeScales: scale the moving image derivatives. Use\n
 *    <tt>(MovingImageDerivativeScales 1 1 0)</tt>\n
 *    to penalize deformations in the z-direction. The default value is that
 *    this feature is not used.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT AdvancedImageToImageMetric : public ImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedImageToImageMetric);

  /** Standard class typedefs. */
  using Self = AdvancedImageToImageMetric;
  using Superclass = ImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedImageToImageMetric, ImageToImageMetric);

  /** Constants for the image dimensions. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, TMovingImage::ImageDimension);
  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using MovingImagePointer = typename MovingImageType::Pointer;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using FixedImagePointer = typename FixedImageType::Pointer;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::GradientImageFilterType;
  using typename Superclass::GradientImageFilterPointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using DerivativeValueType = typename DerivativeType::ValueType;
  using typename Superclass::ParametersType;

  using FixedImageMaskSpatialObject2Type = ImageMaskSpatialObject<Self::FixedImageDimension>;
  using MovingImageMaskSpatialObject2Type = ImageMaskSpatialObject<Self::MovingImageDimension>;

  /** Some useful extra typedefs. */
  using FixedImagePixelType = typename FixedImageType::PixelType;
  using MovingImageRegionType = typename MovingImageType::RegionType;
  using MovingImageDerivativeScalesType = FixedArray<double, Self::MovingImageDimension>;

  /** Typedefs for the ImageSampler. */
  using ImageSamplerType = ImageSamplerBase<FixedImageType>;
  using ImageSamplerPointer = typename ImageSamplerType::Pointer;
  using ImageSampleContainerType = typename ImageSamplerType::OutputVectorContainerType;
  using ImageSampleContainerPointer = typename ImageSamplerType::OutputVectorContainerPointer;

  /** Typedefs for Limiter support. */
  using FixedImageLimiterType = LimiterFunctionBase<RealType, FixedImageDimension>;
  using FixedImageLimiterPointer = typename FixedImageLimiterType::Pointer;
  using FixedImageLimiterOutputType = typename FixedImageLimiterType::OutputType;
  using MovingImageLimiterType = LimiterFunctionBase<RealType, MovingImageDimension>;
  using MovingImageLimiterPointer = typename MovingImageLimiterType::Pointer;
  using MovingImageLimiterOutputType = typename MovingImageLimiterType::OutputType;

  /** Advanced transform. */
  using ScalarType = typename TransformType::ScalarType;
  using AdvancedTransformType = AdvancedTransform<ScalarType, FixedImageDimension, MovingImageDimension>;
  using NumberOfParametersType = typename AdvancedTransformType::NumberOfParametersType;

  /** Typedef's for the B-spline transform. */
  using CombinationTransformType = AdvancedCombinationTransform<ScalarType, FixedImageDimension>;
  using BSplineOrder1TransformType = AdvancedBSplineDeformableTransform<ScalarType, FixedImageDimension, 1>;
  using BSplineOrder2TransformType = AdvancedBSplineDeformableTransform<ScalarType, FixedImageDimension, 2>;
  using BSplineOrder3TransformType = AdvancedBSplineDeformableTransform<ScalarType, FixedImageDimension, 3>;
  using BSplineOrder1TransformPointer = typename BSplineOrder1TransformType::Pointer;
  using BSplineOrder2TransformPointer = typename BSplineOrder2TransformType::Pointer;
  using BSplineOrder3TransformPointer = typename BSplineOrder3TransformType::Pointer;

  /** Hessian type; for SelfHessian (experimental feature) */
  using HessianValueType = typename DerivativeType::ValueType;
  using HessianType = vnl_sparse_matrix<HessianValueType>;

  /** Typedefs for multi-threading. */
  using ThreaderType = itk::PlatformMultiThreader;
  using ThreadInfoType = typename ThreaderType::WorkUnitInfo;

  /** Public methods ********************/

  /** Set the transform, of advanced type. */
  virtual void
  SetTransform(AdvancedTransformType * arg)
  {
    this->Superclass::SetTransform(arg);
    if (this->m_AdvancedTransform != arg)
    {
      this->m_AdvancedTransform = arg;
      this->Modified();
    }
  }


  /** Get the advanced transform. */
  const AdvancedTransformType *
  GetTransform() const override
  {
    return this->m_AdvancedTransform.GetPointer();
  }


  /** Set/Get the image sampler. */
  itkSetObjectMacro(ImageSampler, ImageSamplerType);
  virtual ImageSamplerType *
  GetImageSampler() const
  {
    return this->m_ImageSampler.GetPointer();
  }


  /** Inheriting classes can specify whether they use the image sampler functionality;
   * This method allows the user to inspect this setting. */
  itkGetConstMacro(UseImageSampler, bool);

  /** Set/Get the required ratio of valid samples; default 0.25.
   * When less than this ratio*numberOfSamplesTried samples map
   * inside the moving image buffer, an exception will be thrown. */
  itkSetMacro(RequiredRatioOfValidSamples, double);
  itkGetConstMacro(RequiredRatioOfValidSamples, double);

  /** Set/Get the Moving/Fixed limiter. Its thresholds and bounds are set by the metric.
   * Setting a limiter is only mandatory if GetUse{Fixed,Moving}Limiter() returns true. */
  itkSetObjectMacro(MovingImageLimiter, MovingImageLimiterType);
  itkGetConstObjectMacro(MovingImageLimiter, MovingImageLimiterType);
  itkSetObjectMacro(FixedImageLimiter, FixedImageLimiterType);
  itkGetConstObjectMacro(FixedImageLimiter, FixedImageLimiterType);

  /** A percentage that defines how much the gray value range is extended
   * maxlimit = max + LimitRangeRatio * (max - min)
   * minlimit = min - LimitRangeRatio * (max - min)
   * Default: 0.01;
   * If you use a nearest neighbor or linear interpolator,
   * set it to zero and use a hard limiter. */
  itkSetMacro(MovingLimitRangeRatio, double);
  itkGetConstMacro(MovingLimitRangeRatio, double);
  itkSetMacro(FixedLimitRangeRatio, double);
  itkGetConstMacro(FixedLimitRangeRatio, double);

  /** Inheriting classes can specify whether they use the image limiter functionality.
   * This method allows the user to inspect this setting. */
  itkGetConstMacro(UseFixedImageLimiter, bool);
  itkGetConstMacro(UseMovingImageLimiter, bool);

  /** You may specify a scaling vector for the moving image derivatives.
   * If the UseMovingImageDerivativeScales is true, the moving image derivatives
   * are multiplied by the moving image derivative scales (element-wise)
   * You may use this to avoid deformations in the z-dimension, for example,
   * by setting the moving image derivative scales to (1,1,0).
   * This is a rather experimental feature. In most cases you do not need it.
   */
  itkSetMacro(UseMovingImageDerivativeScales, bool);
  itkGetConstMacro(UseMovingImageDerivativeScales, bool);

  itkSetMacro(ScaleGradientWithRespectToMovingImageOrientation, bool);
  itkGetConstMacro(ScaleGradientWithRespectToMovingImageOrientation, bool);

  itkSetMacro(MovingImageDerivativeScales, MovingImageDerivativeScalesType);
  itkGetConstReferenceMacro(MovingImageDerivativeScales, MovingImageDerivativeScalesType);

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation
   * \li Cache the number of transform parameters
   * \li Initialize the image sampler, if used.
   * \li Check if a B-spline interpolator has been set
   * \li Check if an AdvancedTransform has been set
   */
  void
  Initialize() override;

  /** Experimental feature: compute SelfHessian.
   * This base class just returns an identity matrix of the right size.
   */
  virtual void
  GetSelfHessian(const TransformParametersType & parameters, HessianType & H) const;

  /** Set number of threads to use for computations. */
  virtual void
  SetNumberOfWorkUnits(ThreadIdType numberOfThreads);

  /** Switch the function BeforeThreadedGetValueAndDerivative on or off. */
  itkSetMacro(UseMetricSingleThreaded, bool);
  itkGetConstReferenceMacro(UseMetricSingleThreaded, bool);
  itkBooleanMacro(UseMetricSingleThreaded);

  /** Select the use of multi-threading*/
  // \todo: maybe these can be united, check base class.
  itkSetMacro(UseMultiThread, bool);
  itkGetConstReferenceMacro(UseMultiThread, bool);
  itkBooleanMacro(UseMultiThread);

  /** Contains calls from GetValueAndDerivative that are thread-unsafe,
   * together with preparation for multi-threading.
   * Note that the only reason why this function is not protected, is
   * because the ComboMetric needs to call it.
   */
  virtual void
  BeforeThreadedGetValueAndDerivative(const TransformParametersType & parameters) const;

protected:
  /** Constructor. */
  AdvancedImageToImageMetric();

  /** Destructor. */
  ~AdvancedImageToImageMetric() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Protected Typedefs ******************/

  /** Typedefs for indices and points. */
  using FixedImageIndexType = typename FixedImageType::IndexType;
  using FixedImageIndexValueType = typename FixedImageIndexType::IndexValueType;
  using MovingImageIndexType = typename MovingImageType::IndexType;
  using FixedImagePointType = typename TransformType::InputPointType;
  using MovingImagePointType = typename TransformType::OutputPointType;
  using MovingImageContinuousIndexType = typename InterpolatorType::ContinuousIndexType;

  /** Typedefs used for computing image derivatives. */
  using BSplineInterpolatorType =
    BSplineInterpolateImageFunction<MovingImageType, CoordinateRepresentationType, double>;
  using BSplineInterpolatorPointer = typename BSplineInterpolatorType::Pointer;
  using BSplineInterpolatorFloatType =
    BSplineInterpolateImageFunction<MovingImageType, CoordinateRepresentationType, float>;
  using BSplineInterpolatorFloatPointer = typename BSplineInterpolatorFloatType::Pointer;
  using ReducedBSplineInterpolatorType =
    ReducedDimensionBSplineInterpolateImageFunction<MovingImageType, CoordinateRepresentationType, double>;
  using ReducedBSplineInterpolatorPointer = typename ReducedBSplineInterpolatorType::Pointer;
  using LinearInterpolatorType = AdvancedLinearInterpolateImageFunction<MovingImageType, CoordinateRepresentationType>;
  using LinearInterpolatorPointer = typename LinearInterpolatorType::Pointer;
  using MovingImageDerivativeType = typename BSplineInterpolatorType::CovariantVectorType;
  using CentralDifferenceGradientFilterType = GradientImageFilter<MovingImageType, RealType, RealType>;
  using CentralDifferenceGradientFilterPointer = typename CentralDifferenceGradientFilterType::Pointer;

  /** Typedefs for support of sparse Jacobians and compact support of transformations. */
  using NonZeroJacobianIndicesType = typename AdvancedTransformType::NonZeroJacobianIndicesType;

  /** Protected Variables **************/

  /** Variables for ImageSampler support. m_ImageSampler is mutable,
   * because it is changed in the GetValue(), etc, which are const functions.
   */
  mutable ImageSamplerPointer m_ImageSampler{ nullptr };

  /** Variables for image derivative computation. */
  bool                              m_InterpolatorIsLinear{ false };
  bool                              m_InterpolatorIsBSpline{ false };
  bool                              m_InterpolatorIsBSplineFloat{ false };
  bool                              m_InterpolatorIsReducedBSpline{ false };
  LinearInterpolatorPointer         m_LinearInterpolator{ nullptr };
  BSplineInterpolatorPointer        m_BSplineInterpolator{ nullptr };
  BSplineInterpolatorFloatPointer   m_BSplineInterpolatorFloat{ nullptr };
  ReducedBSplineInterpolatorPointer m_ReducedBSplineInterpolator{ nullptr };

  CentralDifferenceGradientFilterPointer m_CentralDifferenceGradientFilter{ nullptr };

  /** Variables to store the AdvancedTransform. */
  bool                                    m_TransformIsAdvanced{ false };
  typename AdvancedTransformType::Pointer m_AdvancedTransform{ nullptr };
  mutable bool                            m_TransformIsBSpline{ false };

  /** Variables for the Limiters. */
  FixedImageLimiterPointer     m_FixedImageLimiter{ nullptr };
  MovingImageLimiterPointer    m_MovingImageLimiter{ nullptr };
  FixedImagePixelType          m_FixedImageTrueMin{ 0 };
  FixedImagePixelType          m_FixedImageTrueMax{ 1 };
  MovingImagePixelType         m_MovingImageTrueMin{ 0 };
  MovingImagePixelType         m_MovingImageTrueMax{ 1 };
  FixedImageLimiterOutputType  m_FixedImageMinLimit{ 0 };
  FixedImageLimiterOutputType  m_FixedImageMaxLimit{ 1 };
  MovingImageLimiterOutputType m_MovingImageMinLimit{ 0 };
  MovingImageLimiterOutputType m_MovingImageMaxLimit{ 1 };

  /** Multi-threaded metric computation. */

  /** Multi-threaded version of GetValue(). */
  virtual inline void
  ThreadedGetValue(ThreadIdType threadID)
  {}

  /** Finalize multi-threaded metric computation. */
  virtual inline void
  AfterThreadedGetValue(MeasureType & value) const
  {}

  /** GetValue threader callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  GetValueThreaderCallback(void * arg);

  /** Launch MultiThread GetValue. */
  void
  LaunchGetValueThreaderCallback() const;

  /** Multi-threaded version of GetValueAndDerivative(). */
  virtual inline void
  ThreadedGetValueAndDerivative(ThreadIdType threadID)
  {}

  /** Finalize multi-threaded metric computation. */
  virtual inline void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const
  {}

  /** GetValueAndDerivative threader callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  GetValueAndDerivativeThreaderCallback(void * arg);

  /** Launch MultiThread GetValueAndDerivative. */
  void
  LaunchGetValueAndDerivativeThreaderCallback() const;

  /** AccumulateDerivatives threader callback function. */
  static ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
  AccumulateDerivativesThreaderCallback(void * arg);

  /** Variables for multi-threading. */
  bool m_UseMetricSingleThreaded{ true };
  bool m_UseMultiThread{ false };
  bool m_UseOpenMP;

  /** Helper structs that multi-threads the computation of
   * the metric derivative using ITK threads.
   */
  struct MultiThreaderParameterType
  {
    // To give the threads access to all members.
    AdvancedImageToImageMetric * st_Metric;
    // Used for accumulating derivatives
    DerivativeValueType * st_DerivativePointer;
    DerivativeValueType   st_NormalizationFactor;
  };
  mutable MultiThreaderParameterType m_ThreaderMetricParameters;

  /** Most metrics will perform multi-threading by letting
   * each thread compute a part of the value and derivative.
   *
   * These parameters are initialized at every call of GetValueAndDerivative
   * in the function InitializeThreadingParameters(). Since GetValueAndDerivative
   * is const, also InitializeThreadingParameters should be const, and therefore
   * these member variables are mutable.
   */

  // test per thread struct with padding and alignment
  struct GetValuePerThreadStruct
  {
    SizeValueType st_NumberOfPixelsCounted;
    MeasureType   st_Value;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, GetValuePerThreadStruct, PaddedGetValuePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT, PaddedGetValuePerThreadStruct, AlignedGetValuePerThreadStruct);
  mutable std::unique_ptr<AlignedGetValuePerThreadStruct[]> m_GetValuePerThreadVariables{ nullptr };
  mutable ThreadIdType                                      m_GetValuePerThreadVariablesSize{ 0 };

  // test per thread struct with padding and alignment
  struct GetValueAndDerivativePerThreadStruct
  {
    SizeValueType  st_NumberOfPixelsCounted;
    MeasureType    st_Value;
    DerivativeType st_Derivative;
  };
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT,
               GetValueAndDerivativePerThreadStruct,
               PaddedGetValueAndDerivativePerThreadStruct);
  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT,
                    PaddedGetValueAndDerivativePerThreadStruct,
                    AlignedGetValueAndDerivativePerThreadStruct);
  mutable std::unique_ptr<AlignedGetValueAndDerivativePerThreadStruct[]> m_GetValueAndDerivativePerThreadVariables{
    nullptr
  };
  mutable ThreadIdType m_GetValueAndDerivativePerThreadVariablesSize{ 0 };

  /** Initialize some multi-threading related parameters. */
  virtual void
  InitializeThreadingParameters() const;

  /** Protected methods ************** */

  /** Methods for image sampler support **********/

  /** Initialize variables related to the image sampler; called by Initialize. */
  virtual void
  InitializeImageSampler();

  /** Inheriting classes can specify whether they use the image sampler functionality
   * Make sure to set it before calling Initialize; default: false. */
  itkSetMacro(UseImageSampler, bool);

  /** Check if enough samples have been found to compute a reliable
   * estimate of the value/derivative; throws an exception if not. */
  virtual void
  CheckNumberOfSamples(unsigned long wanted, unsigned long found) const;

  /** Methods for image derivative evaluation support **********/

  /** Initialize variables for image derivative computation; this
   * method is called by Initialize. */
  virtual void
  CheckForBSplineInterpolator();

  /** Compute the image value (and possibly derivative) at a transformed point.
   * Checks if the point lies within the moving image buffer (bool return).
   * If no gradient is wanted, set the gradient argument to 0.
   * If a BSplineInterpolationFunction or AdvacnedLinearInterpolationFunction
   * is used, this class obtains image derivatives from the B-spline or linear
   * interpolator. Otherwise, image derivatives are computed using nearest
   * neighbor interpolation of a precomputed (central difference) gradient image.
   */
  virtual bool
  EvaluateMovingImageValueAndDerivative(const MovingImagePointType & mappedPoint,
                                        RealType &                   movingImageValue,
                                        MovingImageDerivativeType *  gradient) const
  {
    return EvaluateMovingImageValueAndDerivativeWithOptionalThreadId(mappedPoint, movingImageValue, gradient);
  }

  /* A faster version of `EvaluateMovingImageValueAndDerivative`: Non-virtual, using multithreading, and doing less
   * dynamic memory allocation/decallocation operations, internally. */
  bool
  FastEvaluateMovingImageValueAndDerivative(const MovingImagePointType & mappedPoint,
                                            RealType &                   movingImageValue,
                                            MovingImageDerivativeType *  gradient,
                                            const ThreadIdType           threadId) const
  {
    return EvaluateMovingImageValueAndDerivativeWithOptionalThreadId(mappedPoint, movingImageValue, gradient, threadId);
  }

  /** Computes the inner product of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns).
   */
  virtual void
  EvaluateTransformJacobianInnerProduct(const TransformJacobianType &     jacobian,
                                        const MovingImageDerivativeType & movingImageDerivative,
                                        DerivativeType &                  imageJacobian) const;

  /** Methods to support transforms with sparse Jacobians, like the BSplineTransform **********/

  /** Check if the transform is an AdvancedTransform. Called by Initialize.
   * If so, we can speed up derivative calculations by only inspecting
   * the parameters in the support region of a point.
   */
  virtual void
  CheckForAdvancedTransform();

  /** Check if the transform is a B-spline. Called by Initialize. */
  virtual void
  CheckForBSplineTransform() const;

  /** Transform a point from FixedImage domain to MovingImage domain. */
  MovingImagePointType
  TransformPoint(const FixedImagePointType & fixedImagePoint) const;

  /** This function returns a reference to the transform Jacobians.
   * This is either a reference to the full TransformJacobian or
   * a reference to a sparse Jacobians.
   * The m_NonZeroJacobianIndices contains the indices that are nonzero.
   * The length of NonZeroJacobianIndices is set in the CheckForAdvancedTransform
   * function. */
  virtual bool
  EvaluateTransformJacobian(const FixedImagePointType &  fixedImagePoint,
                            TransformJacobianType &      jacobian,
                            NonZeroJacobianIndicesType & nzji) const;

  /** Convenience method: check if point is inside the moving mask. *****************/
  virtual bool
  IsInsideMovingMask(const MovingImagePointType & point) const;

  /** Initialize the {Fixed,Moving}[True]{Max,Min}[Limit] and the {Fixed,Moving}ImageLimiter
   * Only does something when Use{Fixed,Moving}Limiter is set to true; */
  virtual void
  InitializeLimiters();

  /** Inheriting classes can specify whether they use the image limiter functionality
   * Make sure to set it before calling Initialize; default: false. */
  itkSetMacro(UseFixedImageLimiter, bool);
  itkSetMacro(UseMovingImageLimiter, bool);

  double m_FixedLimitRangeRatio{ 0.01 };
  double m_MovingLimitRangeRatio{ 0.01 };

private:
  template <typename... TOptionalThreadId>
  bool
  EvaluateMovingImageValueAndDerivativeWithOptionalThreadId(const MovingImagePointType & mappedPoint,
                                                            RealType &                   movingImageValue,
                                                            MovingImageDerivativeType *  gradient,
                                                            const TOptionalThreadId... optionalThreadId) const;

  /** Private member variables. */
  bool   m_UseImageSampler{ false };
  bool   m_UseFixedImageLimiter{ false };
  bool   m_UseMovingImageLimiter{ false };
  double m_RequiredRatioOfValidSamples{ 0.25 };
  bool   m_UseMovingImageDerivativeScales{ false };
  bool   m_ScaleGradientWithRespectToMovingImageOrientation{ false };

  MovingImageDerivativeScalesType m_MovingImageDerivativeScales{ MovingImageDerivativeScalesType::Filled(1.0) };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkAdvancedImageToImageMetric_h
