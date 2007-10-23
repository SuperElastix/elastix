#ifndef __itkKappaStatisticImageToImageMetric2_h
#define __itkKappaStatisticImageToImageMetric2_h

#include "itkAdvancedImageToImageMetric.h"

namespace itk
{
/** \class KappaStatisticImageToImageMetric2
 * \brief Computes similarity between two binary objects to be 
 * registered
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
 * \ingroup RegistrationMetrics
 */
template < class TFixedImage, class TMovingImage > 
class ITK_EXPORT KappaStatisticImageToImageMetric2 : 
    public AdvancedImageToImageMetric< TFixedImage, TMovingImage>
{
public:

  /** Standard class typedefs. */
  typedef KappaStatisticImageToImageMetric2   Self;
  typedef AdvancedImageToImageMetric<
    TFixedImage, TMovingImage >               Superclass;
  typedef SmartPointer<Self>                  Pointer;
  typedef SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );
 
  /** Run-time type information (and related methods). */
  itkTypeMacro( KappaStatisticImageToImageMetric2, AdvancedImageToImageMetric );
 
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

  /** Computes the gradient image and assigns it to m_GradientImage *
  void ComputeGradient();

  /** Get the value of the metric at a particular parameter
   *  setting.  The metric value is given by 2*|A&B|/(|A|+|B|), where A 
   *  is the moving image, B is the fixed image, & is intersection,
   *  and |.| indicates the area of the enclosed set. If ComplementOn has
   *  been set, the metric value is 1.0-2*|A&B|/(|A|+|B|). */
  MeasureType GetValue( const TransformParametersType & parameters ) const;

  /** Get the derivatives of the match measure. This method internally calls
   * the \c GetValueAndDerivative() method. */
  void GetDerivative( const TransformParametersType &,
    DerivativeType & derivative ) const;

  /** Get both the value and derivative. */
  void GetValueAndDerivative( const TransformParametersType & parameters,
    MeasureType& Value, DerivativeType& Derivative ) const;

  /** This method allows the user to set the foreground value.  The default 
   *  value is 1.0. */
  itkSetMacro( ForegroundValue, RealType ); 
  itkGetMacro( ForegroundValue, RealType );

  /** Set/Get whether this metric returns 2*|A&B|/(|A|+|B|) 
   * (ComplementOff, the default) or 1.0 - 2*|A&B|/(|A|+|B|) 
   * (ComplementOn). When using an optimizer that minimizes
   * metric values use ComplementOn().  */
  itkSetMacro( Complement, bool );
  itkBooleanMacro( Complement );
  itkGetMacro( Complement, bool );

protected:
  KappaStatisticImageToImageMetric2();
  virtual ~KappaStatisticImageToImageMetric2() {};
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  KappaStatisticImageToImageMetric2(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  RealType   m_ForegroundValue;
  bool       m_Complement;

}; // end class KappaStatisticImageToImageMetric2

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkKappaStatisticImageToImageMetric2.txx"
#endif

#endif // end #ifndef __itkKappaStatisticImageToImageMetric2_h

