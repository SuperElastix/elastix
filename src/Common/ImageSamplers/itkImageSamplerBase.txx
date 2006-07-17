#ifndef __ImageSamplerBase_txx
#define __ImageSamplerBase_txx

#include "itkImageSamplerBase.h"

namespace itk
{

  /**
	 * ******************* Constructor *******************
	 */

  template< class TInputImage >
    ImageSamplerBase< TInputImage >
    ::ImageSamplerBase()
  {
    this->m_Mask = 0;
  } // end Constructor


  /**
	 * ******************* GenerateInputRequestedRegion *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::GenerateInputRequestedRegion( void )
  {
    /** Check if input image was set. */
    if ( this->GetNumberOfInputs() != 1 )
    {
      itkExceptionMacro( << "ERROR: Input image not set" );
      return;
    }

    /** Get a pointer to the input image. */
    InputImagePointer inputImage = const_cast< InputImageType * >( this->GetInput() );

    /** Get and set the region. */
    if ( this->GetInputImageRegion().GetNumberOfPixels() != 0 )
    {
      InputImageRegionType inputRequestedRegion = this->GetInputImageRegion();

      /** crop the input requested region at the input's largest possible region. */
      if ( inputRequestedRegion.Crop( inputImage->GetLargestPossibleRegion() ) )
      {
        inputImage->SetRequestedRegion( inputRequestedRegion );
        return;
      }
      else
      {
        /** Couldn't crop the region (requested region is outside the largest
         * possible region).  Throw an exception. */

        /** store what we tried to request (prior to trying to crop). */
        inputImage->SetRequestedRegion( inputRequestedRegion );

        /** build an exception. */
        InvalidRequestedRegionError e(__FILE__, __LINE__);
        e.SetLocation(ITK_LOCATION);
        e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
        e.SetDataObject(inputImage);
        throw e;
      } 
    }
    else
    {
      inputImage->SetRequestedRegion( inputImage->GetLargestPossibleRegion() );
      this->SetInputImageRegion( inputImage->GetLargestPossibleRegion() );
    }

  } // end GenerateInputRequestedRegion


  /**
	 * ******************* SelectNewSamplesOnUpdate *******************
	 */
  
  template< class TInputImage >
    bool
    ImageSamplerBase< TInputImage >
    ::SelectNewSamplesOnUpdate( void )
  {
    /** The default behaviour is to select new samples after every update. */
    this->Modified();
    return true;

  } // end SelectNewSamplesOnUpdate


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "NumberOfSamples: " << this->m_NumberOfSamples << std::endl;
    os << indent << "Mask: " << this->m_Mask << std::endl;
  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __ImageSamplerBase_txx

