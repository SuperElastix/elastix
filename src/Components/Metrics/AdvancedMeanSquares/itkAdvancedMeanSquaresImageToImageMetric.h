
#ifndef __itkAdvancedMeanSquaresImageToImageMetric_h
#define __itkAdvancedMeanSquaresImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"

namespace itk
{

/** \class AdvancedMeanSquaresImageToImageMetric
 * \brief Computes similarity between two objects to be registered
 *
 * This Class is templated over the type of the fixed and moving
 * images to be compared.
 *
 * This metric computes the sum of squared differenced between pixels in
 * the moving image and pixels in the fixed image. The spatial correspondance 
 * between both images is established through a Transform. Pixel values are
 * taken from the Moving image. Their positions are mapped to the Fixed image
 * and result in general in non-grid position on it. Values at these non-grid
 * position of the Fixed image are interpolated using a user-selected Interpolator.
 * 
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */

template < class TFixedImage, class TMovingImage > 
class AdvancedMeanSquaresImageToImageMetric : 
    public AdvancedImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef AdvancedMeanSquaresImageToImageMetric		Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >                   Superclass;
  typedef SmartPointer<Self>                      Pointer;
  typedef SmartPointer<const Self>                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedMeanSquaresImageToImageMetric, AdvancedImageToImageMetric );

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
  typedef typename Superclass::ParametersType             ParametersType;
  typedef typename Superclass::FixedImagePixelType        FixedImagePixelType;
  typedef typename Superclass::MovingImageRegionType      MovingImageRegionType;
  typedef typename Superclass::ImageSamplerType           ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer        ImageSamplerPointer;
  typedef typename Superclass::ImageSampleContainerType   ImageSampleContainerType;
  typedef typename 
    Superclass::ImageSampleContainerPointer               ImageSampleContainerPointer;
  typedef typename Superclass::InternalMaskPixelType      InternalMaskPixelType;
  typedef typename
    Superclass::InternalMovingImageMaskType               InternalMovingImageMaskType;
  typedef typename 
    Superclass::MovingImageMaskInterpolatorType           MovingImageMaskInterpolatorType;
  typedef typename Superclass::FixedImageLimiterType      FixedImageLimiterType;
  typedef typename Superclass::MovingImageLimiterType     MovingImageLimiterType;
  typedef typename
    Superclass::FixedImageLimiterOutputType               FixedImageLimiterOutputType;
  typedef typename
    Superclass::MovingImageLimiterOutputType              MovingImageLimiterOutputType;

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
  virtual void GetValueAndDerivative( const TransformParametersType & parameters,
		MeasureType& Value, DerivativeType& Derivative ) const;

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly.
   * \li Call the superclass' implementation
   * \li Estimate the normalization factor, if asked for.  */
  virtual void Initialize(void) throw ( ExceptionObject );

  /** Set/Get whether to normalize the mean squares measure.
   * This divides the MeanSquares by a factor (range/10)^2,
   * where range represents the maximum gray value range of the
   * images. Based on the ad hoc assumption that range/10 is the
   * maximum average difference that will be observed. 
   * Dividing by range^2 sounds less ad hoc, but will yield
   * very small values. */
  itkSetMacro(UseNormalization, bool);
  itkGetConstMacro(UseNormalization, bool);
   
protected:
  AdvancedMeanSquaresImageToImageMetric();
  virtual ~AdvancedMeanSquaresImageToImageMetric() {};
	void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Protected Typedefs ******************/

  /** Typedefs inherited from superclass */
  typedef typename Superclass::FixedImageIndexType                FixedImageIndexType;
	typedef typename Superclass::FixedImageIndexValueType           FixedImageIndexValueType;
	typedef typename Superclass::MovingImageIndexType               MovingImageIndexType;
	typedef typename Superclass::FixedImagePointType                FixedImagePointType;
	typedef typename Superclass::MovingImagePointType               MovingImagePointType;
  typedef typename Superclass::MovingImageContinuousIndexType     MovingImageContinuousIndexType;
  typedef	typename Superclass::BSplineInterpolatorType            BSplineInterpolatorType;
  typedef typename Superclass::ForwardDifferenceFilterType        ForwardDifferenceFilterType;
  typedef typename Superclass::MovingImageDerivativeType          MovingImageDerivativeType;
  typedef typename Superclass::BSplineTransformType               BSplineTransformType;
  typedef typename Superclass::BSplineTransformWeightsType        BSplineTransformWeightsType;
	typedef typename Superclass::BSplineTransformIndexArrayType     BSplineTransformIndexArrayType;
	typedef typename Superclass::BSplineCombinationTransformType    BSplineCombinationTransformType;
 	typedef typename Superclass::BSplineParametersOffsetType        BSplineParametersOffsetType;
  typedef typename Superclass::ParameterIndexArrayType            ParameterIndexArrayType;
  typedef typename Superclass::MovingImageMaskDerivativeType      MovingImageMaskDerivativeType;

  double m_NormalizationFactor;
     
  /** Computes the innerproduct of transform jacobian with moving image gradient
   * and transform jacobian with the derivative of the movingMask
   * The results are stored in imageJacobian and maskJacobian, which are supposed
   * to have the right size (same length as jacobian's number of columns). */
  void EvaluateTransformJacobianInnerProducts(
    const TransformJacobianType & jacobian, 
		const MovingImageDerivativeType & movingImageDerivative,
    const MovingImageMaskDerivativeType & movingMaskDerivative,
    DerivativeType & imageJacobian,
    DerivativeType & maskJacobian) const;

  /** Compute a pixel's contribution to the measure and derivatives;
   * Called by GetValueAndDerivative() */
  void UpdateValueAndDerivativeTerms( 
    RealType fixedImageValue,
    RealType movingImageValue,
    RealType movingMaskValue,
    const DerivativeType & imageJacobian,
    const DerivativeType & maskJacobian,
    MeasureType & measure,
    DerivativeType & numderiv,
    DerivativeType & denderiv  ) const;
 
private:
  AdvancedMeanSquaresImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_UseNormalization;

}; // end class AdvancedMeanSquaresImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedMeanSquaresImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkAdvancedMeanSquaresImageToImageMetric_h

