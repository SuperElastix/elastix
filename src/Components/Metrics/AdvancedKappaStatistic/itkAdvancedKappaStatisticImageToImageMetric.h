/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkAdvancedKappaStatisticImageToImageMetric_h
#define __itkAdvancedKappaStatisticImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

namespace itk
{

/** \class AdvancedKappaStatisticImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.  The metric here is designed for matching
 * pixels in two images with the same exact value.  Only one value can
 * be considered (the default is 255) and can be specified with the
 * SetForegroundValue method.  In the computation of the metric, only
 * foreground pixels are considered.  The metric value is given
 * by 2*|A&B|/(|A|+|B|), where A is the foreground region in the moving
 * image, B is the foreground region in the fixed image, & is intersection,
 * and |.| indicates the area of the enclosed set.  The metric is
 * described in "Morphometric Analysis of White Matter Lesions in MR
 * Images: Method and Validation", A. P. Zijdenbos, B. M. Dawant, R. A.
 * Margolin, A. C. Palmer.
 *
 * This metric is especially useful when considering the similarity between
 * binary images.  Given the nature of binary images, a nearest neighbor
 * interpolator is the preferred interpolator.
 *
 * Metric values range from 0.0 (no foreground alignment) to 1.0
 * (perfect foreground alignment).  When dealing with optimizers that can
 * only minimize a metric, use the ComplementOn() method.
 *
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template < class TFixedImage, class TMovingImage >
class AdvancedKappaStatisticImageToImageMetric :
    public AdvancedImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef AdvancedKappaStatisticImageToImageMetric    Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                   Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedKappaStatisticImageToImageMetric, AdvancedImageToImageMetric );

  /** Typedefs from the superclass. */
  typedef typename
    Superclass::CoordinateRepresentationType              CoordinateRepresentationType;
  typedef typename Superclass::MovingImageType            MovingImageType;
  typedef typename Superclass::MovingImagePixelType       MovingImagePixelType;
  typedef typename Superclass::MovingImageConstPointer    MovingImageConstPointer;
  typedef typename Superclass::FixedImageType             FixedImageType;
  typedef typename Superclass::FixedImageConstPointer     FixedImageConstPointer;
  typedef typename Superclass::FixedImageRegionType       FixedImageRegionType;
  typedef typename Superclass::TransformType              TransformType;
  typedef typename Superclass::TransformPointer           TransformPointer;
  typedef typename Superclass::InputPointType             InputPointType;
  typedef typename Superclass::OutputPointType            OutputPointType;
  typedef typename Superclass::TransformParametersType    TransformParametersType;
  typedef typename Superclass::TransformJacobianType      TransformJacobianType;
  typedef typename Superclass::NumberOfParametersType     NumberOfParametersType;
  typedef typename Superclass::InterpolatorType           InterpolatorType;
  typedef typename Superclass::InterpolatorPointer        InterpolatorPointer;
  typedef typename Superclass::RealType                   RealType;
  typedef typename Superclass::GradientPixelType          GradientPixelType;
  typedef typename Superclass::GradientImageType          GradientImageType;
  typedef typename Superclass::GradientImagePointer       GradientImagePointer;
  typedef typename Superclass::GradientImageFilterType    GradientImageFilterType;
  typedef typename Superclass::GradientImageFilterPointer GradientImageFilterPointer;
  typedef typename Superclass::FixedImageMaskType         FixedImageMaskType;
  typedef typename Superclass::FixedImageMaskPointer      FixedImageMaskPointer;
  typedef typename Superclass::MovingImageMaskType        MovingImageMaskType;
  typedef typename Superclass::MovingImageMaskPointer     MovingImageMaskPointer;
  typedef typename Superclass::MeasureType                MeasureType;
  typedef typename Superclass::DerivativeType             DerivativeType;
  typedef typename Superclass::DerivativeValueType        DerivativeValueType;
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageDerivativeScalesType           MovingImageDerivativeScalesType;
  typedef typename Superclass::ThreaderType               ThreaderType;
  typedef typename Superclass::ThreadInfoType             ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    FixedImageType::ImageDimension );

  /** The moving image dimension. */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    MovingImageType::ImageDimension );

  /** Get the value for single valued optimizers. */
  virtual MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. */
  virtual void GetDerivative( const TransformParametersType & parameters,
    DerivativeType & derivative ) const;

  /** Get value and derivatives for multiple valued optimizers. */
  virtual void GetValueAndDerivativeSingleThreaded(
    const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;
  virtual void GetValueAndDerivative(
    const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** Computes the moving gradient image dM/dx. */
  virtual void ComputeGradient( void );

  /** This method allows the user to set the foreground value. The default value is 1.0. */
  itkSetMacro( ForegroundValue, RealType );
  itkGetConstReferenceMacro( ForegroundValue, RealType );

  /** Select which kind of kappa to compute:
   * 1) compare with a foreground value
   * 2) compare if larger than zero
   */
  itkSetMacro( UseForegroundValue, bool );

  /** Set/Get whether this metric returns 2*|A&B|/(|A|+|B|)
   * (ComplementOff, the default) or 1.0 - 2*|A&B|/(|A|+|B|)
   * (ComplementOn). When using an optimizer that minimizes
   * metric values use ComplementOn().
   */
  itkSetMacro( Complement, bool );
  itkGetConstReferenceMacro( Complement, bool );
  itkBooleanMacro( Complement );

  /** Set the precision. */
  itkSetMacro( Epsilon, RealType );
  itkGetConstReferenceMacro( Epsilon, RealType );

protected:
  AdvancedKappaStatisticImageToImageMetric();
  virtual ~AdvancedKappaStatisticImageToImageMetric();

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
  typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
  typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
  typedef typename Superclass::FixedImagePointType                FixedImagePointType;
  typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  typedef typename Superclass::BSplineInterpolatorType            BSplineInterpolatorType;
  typedef typename Superclass::CentralDifferenceGradientFilterType  CentralDifferenceGradientFilterType;
  typedef typename Superclass::MovingImageDerivativeType          MovingImageDerivativeType;
  typedef typename Superclass::NonZeroJacobianIndicesType         NonZeroJacobianIndicesType;

  /** Compute a pixel's contribution to the measure and derivatives;
   * Called by GetValueAndDerivative().
   */
  void UpdateValueAndDerivativeTerms(
    const RealType & fixedImageValue,
    const RealType & movingImageValue,
    std::size_t & fixedForegroundArea,
    std::size_t & movingForegroundArea,
    std::size_t & intersection,
    const DerivativeType & imageJacobian,
    const NonZeroJacobianIndicesType & nzji,
    DerivativeType & sum1,
    DerivativeType & sum2 ) const;

  /** Initialize some multi-threading related parameters.
   * Overrides function in AdvancedImageToImageMetric, because
   * here we use other parameters.
   */
  virtual void InitializeThreadingParameters( void ) const;

  /** Get value and derivatives for each thread. */
  inline void ThreadedGetValueAndDerivative( ThreadIdType threadID );

  /** Gather the values and derivatives from all threads */
  inline void AfterThreadedGetValueAndDerivative(
    MeasureType & value, DerivativeType & derivative ) const;

  /** AccumulateDerivatives threader callback function */
  static ITK_THREAD_RETURN_TYPE AccumulateDerivativesThreaderCallback( void * arg );

private:
  AdvancedKappaStatisticImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool      m_UseForegroundValue;
  RealType  m_ForegroundValue;
  RealType  m_Epsilon;
  bool      m_Complement;

  /** Threading related parameters. */

  /** Helper structs that multi-threads the computation of
   * the metric derivative using ITK threads.
   */
  struct MultiThreaderAccumulateDerivativeType
  {
    AdvancedKappaStatisticImageToImageMetric * st_Metric;

    MeasureType           st_Coefficient1;
    MeasureType           st_Coefficient2;
    DerivativeValueType * st_DerivativePointer;
  };

  struct KappaGetValueAndDerivativePerThreadStruct
  {
    SizeValueType     st_NumberOfPixelsCounted;
    SizeValueType     st_AreaSum;
    SizeValueType     st_AreaIntersection;
    DerivativeType    st_DerivativeSum1;
    DerivativeType    st_DerivativeSum2;
    TransformJacobianType st_TransformJacobian;
  };
  itkPadStruct( ITK_CACHE_LINE_ALIGNMENT, KappaGetValueAndDerivativePerThreadStruct,
    PaddedKappaGetValueAndDerivativePerThreadStruct );
  itkAlignedTypedef( ITK_CACHE_LINE_ALIGNMENT, PaddedKappaGetValueAndDerivativePerThreadStruct,
    AlignedKappaGetValueAndDerivativePerThreadStruct );
  mutable AlignedKappaGetValueAndDerivativePerThreadStruct * m_KappaGetValueAndDerivativePerThreadVariables;
  mutable ThreadIdType m_KappaGetValueAndDerivativePerThreadVariablesSize;

}; // end class AdvancedKappaStatisticImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedKappaStatisticImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkAdvancedKappaStatisticImageToImageMetric_h

