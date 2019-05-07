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
#ifndef __itkAdvancedImageToImageMetric_h
#define __itkAdvancedImageToImageMetric_h

#include "itkImageToImageMetric.h"

#include "itkImageSamplerBase.h"
#include "itkGradientImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkReducedDimensionBSplineInterpolateImageFunction.h"
#include "itkAdvancedLinearInterpolateImageFunction.h"
#include "itkLimiterFunctionBase.h"
#include "itkFixedArray.h"
#include "itkAdvancedTransform.h"
#include "vnl/vnl_sparse_matrix.h"

#include "itkImageMaskSpatialObject.h"

// Needed for checking for B-spline for faster implementation
#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkAdvancedCombinationTransform.h"

#include "itkMultiThreader.h"

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

template< class TFixedImage, class TMovingImage >
class AdvancedImageToImageMetric :
  public ImageToImageMetric< TFixedImage, TMovingImage >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedImageToImageMetric                      Self;
  typedef ImageToImageMetric< TFixedImage, TMovingImage > Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedImageToImageMetric, ImageToImageMetric );

  /** Constants for the image dimensions. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    TMovingImage::ImageDimension );
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType              MovingImageType;
  typedef typename Superclass::MovingImagePixelType         MovingImagePixelType;
  typedef typename MovingImageType::Pointer                 MovingImagePointer;
  typedef typename Superclass::MovingImageConstPointer      MovingImageConstPointer;
  typedef typename Superclass::FixedImageType               FixedImageType;
  typedef typename FixedImageType::Pointer                  FixedImagePointer;
  typedef typename Superclass::FixedImageConstPointer       FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType         FixedImageRegionType;
  typedef typename Superclass::TransformType                TransformType;
  typedef typename Superclass::TransformPointer             TransformPointer;
  typedef typename Superclass::InputPointType               InputPointType;
  typedef typename Superclass::OutputPointType              OutputPointType;
  typedef typename Superclass::TransformParametersType      TransformParametersType;
  typedef typename Superclass::TransformJacobianType        TransformJacobianType;
  typedef typename Superclass::InterpolatorType             InterpolatorType;
  typedef typename Superclass::InterpolatorPointer          InterpolatorPointer;
  typedef typename Superclass::RealType                     RealType;
  typedef typename Superclass::GradientPixelType            GradientPixelType;
  typedef typename Superclass::GradientImageType            GradientImageType;
  typedef typename Superclass::GradientImagePointer         GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType      GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer   GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType           FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer        FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType          MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer       MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                  MeasureType;
  typedef typename Superclass::DerivativeType               DerivativeType;
  typedef typename DerivativeType::ValueType                DerivativeValueType;
  typedef typename Superclass::ParametersType               ParametersType;

  typedef ImageMaskSpatialObject< itkGetStaticConstMacro( FixedImageDimension ) > FixedImageMaskSpatialObject2Type;
  typedef ImageMaskSpatialObject< itkGetStaticConstMacro( MovingImageDimension ) > MovingImageMaskSpatialObject2Type;

  /** Some useful extra typedefs. */
  typedef typename FixedImageType::PixelType               FixedImagePixelType;
  typedef typename MovingImageType::RegionType             MovingImageRegionType;
  typedef FixedArray< double, Self::MovingImageDimension > MovingImageDerivativeScalesType;

  /** Typedefs for the ImageSampler. */
  typedef ImageSamplerBase< FixedImageType >                      ImageSamplerType;
  typedef typename ImageSamplerType::Pointer                      ImageSamplerPointer;
  typedef typename ImageSamplerType::OutputVectorContainerType    ImageSampleContainerType;
  typedef typename ImageSamplerType::OutputVectorContainerPointer ImageSampleContainerPointer;

  /** Typedefs for Limiter support. */
  typedef LimiterFunctionBase< RealType, FixedImageDimension >  FixedImageLimiterType;
  typedef typename FixedImageLimiterType::Pointer               FixedImageLimiterPointer;
  typedef typename FixedImageLimiterType::OutputType            FixedImageLimiterOutputType;
  typedef LimiterFunctionBase< RealType, MovingImageDimension > MovingImageLimiterType;
  typedef typename MovingImageLimiterType::Pointer              MovingImageLimiterPointer;
  typedef typename MovingImageLimiterType::OutputType           MovingImageLimiterOutputType;

  /** Advanced transform. */
  typedef typename TransformType::ScalarType ScalarType;
  typedef AdvancedTransform<
    ScalarType, FixedImageDimension, MovingImageDimension >      AdvancedTransformType;
  typedef typename AdvancedTransformType::NumberOfParametersType NumberOfParametersType;

  /** Typedef's for the B-spline transform. */
  typedef AdvancedCombinationTransform< ScalarType, FixedImageDimension >          CombinationTransformType;
  typedef AdvancedBSplineDeformableTransform< ScalarType, FixedImageDimension, 1 > BSplineOrder1TransformType;
  typedef AdvancedBSplineDeformableTransform< ScalarType, FixedImageDimension, 2 > BSplineOrder2TransformType;
  typedef AdvancedBSplineDeformableTransform< ScalarType, FixedImageDimension, 3 > BSplineOrder3TransformType;
  typedef typename BSplineOrder1TransformType::Pointer                             BSplineOrder1TransformPointer;
  typedef typename BSplineOrder2TransformType::Pointer                             BSplineOrder2TransformPointer;
  typedef typename BSplineOrder3TransformType::Pointer                             BSplineOrder3TransformPointer;

  /** Hessian type; for SelfHessian (experimental feature) */
  typedef typename DerivativeType::ValueType    HessianValueType;
  typedef vnl_sparse_matrix< HessianValueType > HessianType;

  /** Typedefs for multi-threading. */
  typedef itk::MultiThreader                      ThreaderType;
  typedef typename ThreaderType::ThreadInfoStruct ThreadInfoType;

  /** Public methods ********************/

  /** Set the transform, of advanced type. */
  virtual void SetTransform( AdvancedTransformType * arg )
  {
    this->Superclass::SetTransform( arg );
    if( this->m_AdvancedTransform != arg )
    {
      this->m_AdvancedTransform = arg;
      this->Modified();
    }
  }


  /** Get the advanced transform. */
  const AdvancedTransformType * GetTransform( void ) const override
  {
    return this->m_AdvancedTransform.GetPointer();
  }


  /** Set/Get the image sampler. */
  itkSetObjectMacro( ImageSampler, ImageSamplerType );
  virtual ImageSamplerType * GetImageSampler( void ) const
  {
    return this->m_ImageSampler.GetPointer();
  }


  /** Inheriting classes can specify whether they use the image sampler functionality;
   * This method allows the user to inspect this setting. */
  itkGetConstMacro( UseImageSampler, bool );

  /** Set/Get the required ratio of valid samples; default 0.25.
   * When less than this ratio*numberOfSamplesTried samples map
   * inside the moving image buffer, an exception will be thrown. */
  itkSetMacro( RequiredRatioOfValidSamples, double );
  itkGetConstMacro( RequiredRatioOfValidSamples, double );

  /** Set/Get the Moving/Fixed limiter. Its thresholds and bounds are set by the metric.
   * Setting a limiter is only mandatory if GetUse{Fixed,Moving}Limiter() returns true. */
  itkSetObjectMacro( MovingImageLimiter, MovingImageLimiterType );
  itkGetConstObjectMacro( MovingImageLimiter, MovingImageLimiterType );
  itkSetObjectMacro( FixedImageLimiter, FixedImageLimiterType );
  itkGetConstObjectMacro( FixedImageLimiter, FixedImageLimiterType );

  /** A percentage that defines how much the gray value range is extended
   * maxlimit = max + LimitRangeRatio * (max - min)
   * minlimit = min - LimitRangeRatio * (max - min)
   * Default: 0.01;
   * If you use a nearest neighbor or linear interpolator,
   * set it to zero and use a hard limiter. */
  itkSetMacro( MovingLimitRangeRatio, double );
  itkGetConstMacro( MovingLimitRangeRatio, double );
  itkSetMacro( FixedLimitRangeRatio, double );
  itkGetConstMacro( FixedLimitRangeRatio, double );

  /** Inheriting classes can specify whether they use the image limiter functionality.
   * This method allows the user to inspect this setting. */
  itkGetConstMacro( UseFixedImageLimiter, bool );
  itkGetConstMacro( UseMovingImageLimiter, bool );

  /** You may specify a scaling vector for the moving image derivatives.
   * If the UseMovingImageDerivativeScales is true, the moving image derivatives
   * are multiplied by the moving image derivative scales (element-wise)
   * You may use this to avoid deformations in the z-dimension, for example,
   * by setting the moving image derivative scales to (1,1,0).
   * This is a rather experimental feature. In most cases you do not need it.
   */
  itkSetMacro( UseMovingImageDerivativeScales, bool );
  itkGetConstMacro( UseMovingImageDerivativeScales, bool );

  itkSetMacro( ScaleGradientWithRespectToMovingImageOrientation, bool );
  itkGetConstMacro( ScaleGradientWithRespectToMovingImageOrientation, bool );

  itkSetMacro( MovingImageDerivativeScales, MovingImageDerivativeScalesType );
  itkGetConstReferenceMacro( MovingImageDerivativeScales, MovingImageDerivativeScalesType );

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation
   * \li Cache the number of transform parameters
   * \li Initialize the image sampler, if used.
   * \li Check if a B-spline interpolator has been set
   * \li Check if an AdvancedTransform has been set
   */
  void Initialize( void ) override;

  /** Experimental feature: compute SelfHessian.
   * This base class just returns an identity matrix of the right size.
   */
  virtual void GetSelfHessian( const TransformParametersType & parameters, HessianType & H ) const;

  /** Set number of threads to use for computations. */
  virtual void SetNumberOfThreads( ThreadIdType numberOfThreads );

  /** Switch the function BeforeThreadedGetValueAndDerivative on or off. */
  itkSetMacro( UseMetricSingleThreaded, bool );
  itkGetConstReferenceMacro( UseMetricSingleThreaded, bool );
  itkBooleanMacro( UseMetricSingleThreaded );

  /** Select the use of multi-threading*/
  // \todo: maybe these can be united, check base class.
  itkSetMacro( UseMultiThread, bool );
  itkGetConstReferenceMacro( UseMultiThread, bool );
  itkBooleanMacro( UseMultiThread );

  /** Contains calls from GetValueAndDerivative that are thread-unsafe,
   * together with preparation for multi-threading.
   * Note that the only reason why this function is not protected, is
   * because the ComboMetric needs to call it.
   */
  virtual void BeforeThreadedGetValueAndDerivative(
    const TransformParametersType & parameters ) const;

protected:

  /** Constructor. */
  AdvancedImageToImageMetric();

  /** Destructor. */
  ~AdvancedImageToImageMetric() override;

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const override;

  /** Protected Typedefs ******************/

  /** Typedefs for indices and points. */
  typedef typename FixedImageType::IndexType             FixedImageIndexType;
  typedef typename FixedImageIndexType::IndexValueType   FixedImageIndexValueType;
  typedef typename MovingImageType::IndexType            MovingImageIndexType;
  typedef typename TransformType::InputPointType         FixedImagePointType;
  typedef typename TransformType::OutputPointType        MovingImagePointType;
  typedef typename InterpolatorType::ContinuousIndexType MovingImageContinuousIndexType;

  /** Typedefs used for computing image derivatives. */
  typedef BSplineInterpolateImageFunction<
    MovingImageType, CoordinateRepresentationType, double >      BSplineInterpolatorType;
  typedef typename BSplineInterpolatorType::Pointer BSplineInterpolatorPointer;
  typedef BSplineInterpolateImageFunction<
    MovingImageType, CoordinateRepresentationType, float >       BSplineInterpolatorFloatType;
  typedef typename BSplineInterpolatorFloatType::Pointer BSplineInterpolatorFloatPointer;
  typedef ReducedDimensionBSplineInterpolateImageFunction<
    MovingImageType, CoordinateRepresentationType, double >      ReducedBSplineInterpolatorType;
  typedef typename ReducedBSplineInterpolatorType::Pointer ReducedBSplineInterpolatorPointer;
  typedef AdvancedLinearInterpolateImageFunction<
    MovingImageType, CoordinateRepresentationType >              LinearInterpolatorType;
  typedef typename LinearInterpolatorType::Pointer              LinearInterpolatorPointer;
  typedef typename BSplineInterpolatorType::CovariantVectorType MovingImageDerivativeType;
  typedef GradientImageFilter<
    MovingImageType, RealType, RealType >                        CentralDifferenceGradientFilterType;
  typedef typename CentralDifferenceGradientFilterType::Pointer CentralDifferenceGradientFilterPointer;

  /** Typedefs for support of sparse Jacobians and compact support of transformations. */
  typedef typename
    AdvancedTransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Protected Variables **************/

  /** Variables for ImageSampler support. m_ImageSampler is mutable,
   * because it is changed in the GetValue(), etc, which are const functions.
   */
  mutable ImageSamplerPointer m_ImageSampler;

  /** Variables for image derivative computation. */
  bool                                   m_InterpolatorIsLinear;
  bool                                   m_InterpolatorIsBSpline;
  bool                                   m_InterpolatorIsBSplineFloat;
  bool                                   m_InterpolatorIsReducedBSpline;
  LinearInterpolatorPointer              m_LinearInterpolator;
  BSplineInterpolatorPointer             m_BSplineInterpolator;
  BSplineInterpolatorFloatPointer        m_BSplineInterpolatorFloat;
  ReducedBSplineInterpolatorPointer      m_ReducedBSplineInterpolator;

  CentralDifferenceGradientFilterPointer m_CentralDifferenceGradientFilter;

  /** Variables to store the AdvancedTransform. */
  bool m_TransformIsAdvanced;
  typename AdvancedTransformType::Pointer m_AdvancedTransform;
  mutable bool m_TransformIsBSpline;

  /** Variables for the Limiters. */
  FixedImageLimiterPointer     m_FixedImageLimiter;
  MovingImageLimiterPointer    m_MovingImageLimiter;
  FixedImagePixelType          m_FixedImageTrueMin;
  FixedImagePixelType          m_FixedImageTrueMax;
  MovingImagePixelType         m_MovingImageTrueMin;
  MovingImagePixelType         m_MovingImageTrueMax;
  FixedImageLimiterOutputType  m_FixedImageMinLimit;
  FixedImageLimiterOutputType  m_FixedImageMaxLimit;
  MovingImageLimiterOutputType m_MovingImageMinLimit;
  MovingImageLimiterOutputType m_MovingImageMaxLimit;

#if ITK_VERSION_MAJOR >= 5
  /** \note This is a workaround for ITK5, which renamed NumberOfThreads to NumberOfWorkUnits. */
  ThreadIdType GetNumberOfThreads() const
  {
    return Superclass::GetNumberOfWorkUnits();
  }
#endif

  /** Multi-threaded metric computation. */

  /** Multi-threaded version of GetValue(). */
  virtual inline void ThreadedGetValue( ThreadIdType threadID ){}

  /** Finalize multi-threaded metric computation. */
  virtual inline void AfterThreadedGetValue( MeasureType & value ) const {}

  /** GetValue threader callback function. */
  static ITK_THREAD_RETURN_TYPE GetValueThreaderCallback( void * arg );

  /** Launch MultiThread GetValue. */
  void LaunchGetValueThreaderCallback( void ) const;

  /** Multi-threaded version of GetValueAndDerivative(). */
  virtual inline void ThreadedGetValueAndDerivative(
    ThreadIdType threadID ){}

  /** Finalize multi-threaded metric computation. */
  virtual inline void AfterThreadedGetValueAndDerivative(
    MeasureType & value, DerivativeType & derivative ) const {}

  /** GetValueAndDerivative threader callback function. */
  static ITK_THREAD_RETURN_TYPE GetValueAndDerivativeThreaderCallback( void * arg );

  /** Launch MultiThread GetValueAndDerivative. */
  void LaunchGetValueAndDerivativeThreaderCallback( void ) const;

  /** AccumulateDerivatives threader callback function. */
  static ITK_THREAD_RETURN_TYPE AccumulateDerivativesThreaderCallback( void * arg );

  /** Variables for multi-threading. */
  bool m_UseMetricSingleThreaded;
  bool m_UseMultiThread;
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
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, GetValuePerThreadStruct,
    PaddedGetValuePerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedGetValuePerThreadStruct,
    AlignedGetValuePerThreadStruct );
  mutable AlignedGetValuePerThreadStruct * m_GetValuePerThreadVariables;
  mutable ThreadIdType                     m_GetValuePerThreadVariablesSize;

  // test per thread struct with padding and alignment
  struct GetValueAndDerivativePerThreadStruct
  {
    SizeValueType  st_NumberOfPixelsCounted;
    MeasureType    st_Value;
    DerivativeType st_Derivative;
  };
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, GetValueAndDerivativePerThreadStruct,
    PaddedGetValueAndDerivativePerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedGetValueAndDerivativePerThreadStruct,
    AlignedGetValueAndDerivativePerThreadStruct );
  mutable AlignedGetValueAndDerivativePerThreadStruct * m_GetValueAndDerivativePerThreadVariables;
  mutable ThreadIdType                                  m_GetValueAndDerivativePerThreadVariablesSize;

  /** Initialize some multi-threading related parameters. */
  virtual void InitializeThreadingParameters( void ) const;

  /** Protected methods ************** */

  /** Methods for image sampler support **********/

  /** Initialize variables related to the image sampler; called by Initialize. */
  virtual void InitializeImageSampler( void );

  /** Inheriting classes can specify whether they use the image sampler functionality
   * Make sure to set it before calling Initialize; default: false. */
  itkSetMacro( UseImageSampler, bool );

  /** Check if enough samples have been found to compute a reliable
   * estimate of the value/derivative; throws an exception if not. */
  virtual void CheckNumberOfSamples(
    unsigned long wanted, unsigned long found ) const;

  /** Methods for image derivative evaluation support **********/

  /** Initialize variables for image derivative computation; this
   * method is called by Initialize. */
  virtual void CheckForBSplineInterpolator( void );

  /** Compute the image value (and possibly derivative) at a transformed point.
   * Checks if the point lies within the moving image buffer (bool return).
   * If no gradient is wanted, set the gradient argument to 0.
   * If a BSplineInterpolationFunction or AdvacnedLinearInterpolationFunction
   * is used, this class obtains image derivatives from the B-spline or linear
   * interpolator. Otherwise, image derivatives are computed using nearest
   * neighbor interpolation of a precomputed (central difference) gradient image.
   */
  virtual bool EvaluateMovingImageValueAndDerivative(
    const MovingImagePointType & mappedPoint,
    RealType & movingImageValue,
    MovingImageDerivativeType * gradient ) const;

  /** Computes the inner product of transform Jacobian with moving image gradient.
   * The results are stored in imageJacobian, which is supposed
   * to have the right size (same length as Jacobian's number of columns).
   */
  virtual void EvaluateTransformJacobianInnerProduct(
    const TransformJacobianType & jacobian,
    const MovingImageDerivativeType & movingImageDerivative,
    DerivativeType & imageJacobian ) const;

  /** Methods to support transforms with sparse Jacobians, like the BSplineTransform **********/

  /** Check if the transform is an AdvancedTransform. Called by Initialize.
   * If so, we can speed up derivative calculations by only inspecting
   * the parameters in the support region of a point.
   */
  virtual void CheckForAdvancedTransform( void );

  /** Check if the transform is a B-spline. Called by Initialize. */
  virtual void CheckForBSplineTransform( void ) const;

  /** Transform a point from FixedImage domain to MovingImage domain.
   * This function also checks if mapped point is within support region of
   * the transform. It returns true if so, and false otherwise.
   */
  virtual bool TransformPoint(
    const FixedImagePointType & fixedImagePoint,
    MovingImagePointType & mappedPoint ) const;

  /** This function returns a reference to the transform Jacobians.
   * This is either a reference to the full TransformJacobian or
   * a reference to a sparse Jacobians.
   * The m_NonZeroJacobianIndices contains the indices that are nonzero.
   * The length of NonZeroJacobianIndices is set in the CheckForAdvancedTransform
   * function. */
  virtual bool EvaluateTransformJacobian(
    const FixedImagePointType & fixedImagePoint,
    TransformJacobianType & jacobian,
    NonZeroJacobianIndicesType & nzji ) const;

  /** Convenience method: check if point is inside the moving mask. *****************/
  virtual bool IsInsideMovingMask( const MovingImagePointType & point ) const;

  /** Initialize the {Fixed,Moving}[True]{Max,Min}[Limit] and the {Fixed,Moving}ImageLimiter
   * Only does something when Use{Fixed,Moving}Limiter is set to true; */
  virtual void InitializeLimiters( void );

  /** Inheriting classes can specify whether they use the image limiter functionality
   * Make sure to set it before calling Initialize; default: false. */
  itkSetMacro( UseFixedImageLimiter, bool );
  itkSetMacro( UseMovingImageLimiter, bool );

  double m_FixedLimitRangeRatio;
  double m_MovingLimitRangeRatio;

private:

  AdvancedImageToImageMetric( const Self & ); // purposely not implemented
  void operator=( const Self & );             // purposely not implemented

  /** Private member variables. */
  bool   m_UseImageSampler;
  bool   m_UseFixedImageLimiter;
  bool   m_UseMovingImageLimiter;
  double m_RequiredRatioOfValidSamples;
  bool   m_UseMovingImageDerivativeScales;
  bool   m_ScaleGradientWithRespectToMovingImageOrientation;

  MovingImageDerivativeScalesType m_MovingImageDerivativeScales;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkAdvancedImageToImageMetric_h
