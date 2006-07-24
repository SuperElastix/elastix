#ifndef _itkImageToImageMetricWithSampling_txx
#define _itkImageToImageMetricWithSampling_txx

#include "itkImageToImageMetricWithSampling.h"


namespace itk
{

  /**
	 * ********************* Constructor ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    ImageToImageMetricWithSampling<TFixedImage,TMovingImage>
    ::ImageToImageMetricWithSampling()
  {
    this->m_ImageSampler = 0;
  } // end Constructor


  /**
	 * ********************* Initialize ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    ImageToImageMetricWithSampling<TFixedImage,TMovingImage>
    ::Initialize(void) throw ( ExceptionObject )
  {
    /** Initialize transform, interpolator, etc. */
    Superclass::Initialize();

    /** Chech if the ImageSampler is set. */
    if( !this->m_ImageSampler )
    {
      itkExceptionMacro( << "ImageSampler is not present" );
    }

    /** Initialize the Image Sampler. */
    this->m_ImageSampler->SetInput( this->m_FixedImage );
    this->m_ImageSampler->SetMask( this->m_FixedImageMask );
    this->m_ImageSampler->SetInputImageRegion( this->GetFixedImageRegion() );

    // If there are any observers on the metric, call them to give the
    // user code a chance to set parameters on the metric
    this->InvokeEvent( InitializeEvent() );

  } // end Initialize


  /**
	 * ********************* PrintSelf ****************************
	 */

  template <class TFixedImage, class TMovingImage>
    void
    ImageToImageMetricWithSampling<TFixedImage,TMovingImage>
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    Superclass::PrintSelf( os, indent );
    os << indent << "ImageSampler: " << this->m_ImageSampler.GetPointer() << std::endl;
  } // end PrintSelf


} // end namespace itk


#endif // end #ifndef _itkImageToImageMetricWithSampling_txx

