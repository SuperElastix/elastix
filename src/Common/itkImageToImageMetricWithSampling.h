#ifndef __itkImageToImageMetricWithSampling_h
#define __itkImageToImageMetricWithSampling_h

#include "itkImageToImageMetric.h"

#include "itkImageSamplerBase.h"

namespace itk
{
  
/** \class ImageToImageMetricWithSampling
 * \brief Computes similarity between regions of two images.
 *
 * It is possible to set an ImageSampler, which selects the
 * fixed image samples over which the metric is evaluated.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedImage, class TMovingImage>
class ImageToImageMetricWithSampling :
  public ImageToImageMetric< TFixedImage, TMovingImage >
{
public:
  /** Standard class typedefs. */
  typedef ImageToImageMetricWithSampling  Self;
  typedef ImageToImageMetric<
    TFixedImage, TMovingImage >           Superclass;
  typedef SmartPointer<Self>              Pointer;
  typedef SmartPointer<const Self>        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageToImageMetricWithSampling, ImageToImageMetric );

  /** Constants for the image dimensions */
  itkStaticConstMacro( MovingImageDimension, unsigned int,
    TMovingImage::ImageDimension );
  itkStaticConstMacro( FixedImageDimension, unsigned int,
    TFixedImage::ImageDimension );

  /** Typedefs from the superclass. */
  typedef typename Superclass::CoordinateRepresentationType   CoordinateRepresentationType;
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

  /** Typedefs for the ImageSampler. */
  typedef ImageSamplerBase< FixedImageType >              ImageSamplerType;
  typedef typename ImageSamplerType::Pointer              ImageSamplerPointer;
  typedef typename ImageSamplerType::OutputVectorContainerType    ImageSampleContainerType;
  typedef typename ImageSamplerType::OutputVectorContainerPointer ImageSampleContainerPointer;

  /** Set the image sampler. */
  itkSetObjectMacro( ImageSampler, ImageSamplerType );

  /** Get the image sampler. */
  virtual ImageSamplerType * GetImageSampler(void) const
  {
    return this->m_ImageSampler.GetPointer();
  };

  /** Initialize the Metric by making sure that all the components
   *  are present and plugged together correctly. */
  virtual void Initialize(void) throw ( ExceptionObject );

protected:
  ImageToImageMetricWithSampling();
  virtual ~ImageToImageMetricWithSampling() {};
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Member variables. m_ImageSampler is mutable, because it is
   * changed in the GetValue(), etc, which are const function.
   */
  mutable ImageSamplerPointer   m_ImageSampler;

private:
  ImageToImageMetricWithSampling(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
}; // end class ImageToImageMetricWithSampling

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageToImageMetricWithSampling.txx"
#endif

#endif // end #ifndef __itkImageToImageMetricWithSampling_h



