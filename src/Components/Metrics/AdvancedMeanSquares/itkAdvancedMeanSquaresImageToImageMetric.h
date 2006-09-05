
#ifndef __itkAdvancedMeanSquaresImageToImageMetric_h
#define __itkAdvancedMeanSquaresImageToImageMetric_h

#include "itkImageToImageMetricWithSampling.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkImage.h"

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
 * This class provides functionality to calculate (the derivative of) the
 * mean squares metric on only a subset of the fixed image voxels. This
 * option is controlled by the boolean UseAllPixels, which is by default true.
 * Substantial speedup can be accomplished by setting it to false and specifying
 * the NumberOfSpacialSamples to some small portion of the total number of fixed
 * image samples. The samples are randomly chosen using an
 * itk::ImageRandomConstIteratorWithIndex Every iteration a new set of those
 * samples are used. This is important, because the error made by calculating
 * the metric value with only a subset of all samples should be randomly
 * distributed with zero mean.
 *
 * \todo In the while loop in GetValue and GetValueAndDerivative another for
 * loop is made over all parameters. In case of a B-spline transform advantage
 * can be taken from the fact that it has compact support, similar to the
 * itk::MattesMutualInformationImageToImageMetric.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */
template < class TFixedImage, class TMovingImage > 
class AdvancedMeanSquaresImageToImageMetric : 
    public ImageToImageMetricWithSampling< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef AdvancedMeanSquaresImageToImageMetric		Self;
  typedef ImageToImageMetricWithSampling<
    TFixedImage, TMovingImage >             Superclass;
  typedef SmartPointer<Self>                Pointer;
  typedef SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedMeanSquaresImageToImageMetric, ImageToImageMetricWithSampling );

  /** Types transferred from the base class */
  typedef typename Superclass::RealType                 RealType;
  typedef typename Superclass::TransformType            TransformType;
  typedef typename Superclass::TransformPointer         TransformPointer;
  typedef typename Superclass::TransformParametersType  TransformParametersType;
  typedef typename Superclass::TransformJacobianType    TransformJacobianType;
  typedef typename Superclass::GradientPixelType        GradientPixelType;
  typedef typename Superclass::MeasureType              MeasureType;
  typedef typename Superclass::DerivativeType           DerivativeType;
  typedef typename Superclass::FixedImageType           FixedImageType;
  typedef typename Superclass::MovingImageType          MovingImageType;
  typedef typename Superclass::FixedImageConstPointer   FixedImageConstPointer;
  typedef typename Superclass::MovingImageConstPointer  MovingImageConstPointer;
  typedef typename Superclass::InputPointType			      InputPointType;
  typedef typename Superclass::OutputPointType		      OutputPointType;
  typedef typename Superclass::ImageSamplerType         ImageSamplerType;
  typedef typename Superclass::ImageSamplerPointer      ImageSamplerPointer;
  typedef typename 
    Superclass::ImageSampleContainerType                ImageSampleContainerType;
  typedef typename 
    Superclass::ImageSampleContainerPointer             ImageSampleContainerPointer;
  typedef typename Superclass::FixedImageMaskType       FixedImageMaskType;
  typedef typename Superclass::MovingImageMaskType      MovingImageMaskType;
  
  typedef typename 
    Superclass::CoordinateRepresentationType            CoordinateRepresentationType;
 
	/** The fixed image dimension. */
	itkStaticConstMacro( FixedImageDimension, unsigned int,
		FixedImageType::ImageDimension );

	/** The moving image dimension. */
	itkStaticConstMacro( MovingImageDimension, unsigned int,
		MovingImageType::ImageDimension );

  typedef float                                          InternalMaskPixelType;
  typedef typename itk::Image<
    InternalMaskPixelType, 
    itkGetStaticConstMacro(MovingImageDimension) >        InternalMovingImageMaskType;
  typedef typename MovingImageType::SpacingType           MovingImageSpacingType;
  typedef itk::BSplineResampleImageFunction<
    InternalMovingImageMaskType,
    CoordinateRepresentationType >                        MovingImageMaskInterpolatorType;
  typedef typename 
    MovingImageMaskInterpolatorType::CovariantVectorType  MovingImageMaskDerivativeType;
  typedef typename 
    MovingImageMaskInterpolatorType::ContinuousIndexType  MovingImageContinuousIndexType;

  typedef itk::BSplineInterpolateImageFunction<
    MovingImageType,
    CoordinateRepresentationType,
    double>                                               BSplineInterpolatorType;
  
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
   * \li set the internal moving mask
   */
  virtual void Initialize(void) throw ( ExceptionObject );

  /** Get the internal moving image mask. Equals the movingimage mask if set, and 
   * otherwise it's a box with size equal to the moving image's largest possible region */
  itkGetConstObjectMacro(InternalMovingImageMask, InternalMovingImageMaskType);

  /** Get the interpolator of the internal moving image mask */
  itkGetConstObjectMacro(MovingImageMaskInterpolator, MovingImageMaskInterpolatorType);

  /** Set/Get whether the overlap should be taken into account while computing the derivative
   * This setting also affects the value of the metric. Default: true; */
  itkSetMacro(UseDifferentiableOverlap, bool);
  itkGetConstMacro(UseDifferentiableOverlap, bool);

  /** Set the interpolation spline order for the moving image mask; default: 2
   * Make sure to call this before calling Initialize(), if you want to change it. */
  virtual void SetMovingImageMaskInterpolationOrder(unsigned int order)
  {
    this->m_MovingImageMaskInterpolator->SetSplineOrder( order );
  };
  /** Get the interpolation spline order for the moving image mask */
  virtual const unsigned int GetMovingImageMaskInterpolationOrder(void) const
  {
    return this->m_MovingImageMaskInterpolator->GetSplineOrder();
  };
  
protected:
  AdvancedMeanSquaresImageToImageMetric();
  virtual ~AdvancedMeanSquaresImageToImageMetric() {};
	void PrintSelf( std::ostream& os, Indent indent ) const;
  
  /** Estimate value and spatial derivative of internal moving mask */
  virtual void EvaluateMovingMaskValueAndDerivative(
    const OutputPointType & point,
    double & value,
    MovingImageMaskDerivativeType & derivative) const;
  
  /** Estimate value of internal moving mask */
  virtual void EvaluateMovingMaskValue(
    const OutputPointType & point,
    double & value ) const;

  /** Get the moving image value and derivative; if a bspline interpolator is used
   * it is used to compute the derivative. If not, the precomputed GradientImage is used.
   * Returns true if the value and derivative are valid. 
   */
  virtual bool EvaluateMovingImageValueAndDerivative(
    const OutputPointType & point,
    RealType & value,
    GradientPixelType & derivative) const;

  /** Functions called from Initialize, to split up that function a bit. */
  virtual void InitializeInternalMasks(void);

  typename InternalMovingImageMaskType::Pointer      m_InternalMovingImageMask;
  typename MovingImageMaskInterpolatorType::Pointer  m_MovingImageMaskInterpolator;
  typename BSplineInterpolatorType::Pointer          m_BSplineInterpolator;

private:
  AdvancedMeanSquaresImageToImageMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  bool m_UseDifferentiableOverlap;

}; // end class AdvancedMeanSquaresImageToImageMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedMeanSquaresImageToImageMetric.hxx"
#endif

#endif // end #ifndef __itkAdvancedMeanSquaresImageToImageMetric_h

