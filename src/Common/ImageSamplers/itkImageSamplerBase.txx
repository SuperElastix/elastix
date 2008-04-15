/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
    this->m_NumberOfMasks = 0;
    this->m_NumberOfInputImageRegions = 0;

  } // end Constructor()


  /**
	 * ******************* SetMask *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::SetMask( const MaskType *_arg, unsigned int pos )
  {
    if ( this->m_MaskVector.size() < pos + 1 )
    {
      this->m_MaskVector.resize( pos + 1 );
      this->m_NumberOfMasks = pos + 1;
    }
    if ( pos == 0 )
    {
      this->m_Mask = _arg;
    }
    if ( this->m_MaskVector[ pos ] != _arg )
    {
      this->m_MaskVector[ pos ] = _arg;
      this->Modified();
    }

  } // SetMask()


  /**
	 * ******************* GetMask *******************
	 */
  
  template< class TInputImage >
    const typename ImageSamplerBase< TInputImage >::MaskType *
    ImageSamplerBase< TInputImage >
    ::GetMask( unsigned int pos ) const
  {
    if ( this->m_MaskVector.size() < pos + 1 )
    {
      return 0;
    }
    return this->m_MaskVector[ pos ];

  } // end GetMask()


  /**
	 * ******************* SetNumberOfMasks *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::SetNumberOfMasks( const unsigned int _arg )
  {
    if ( this->m_NumberOfMasks != _arg )
    {
      this->m_MaskVector.resize( _arg );
      this->m_NumberOfMasks = _arg;
      this->Modified();
    }

  } // end SetNumberOfMasks()


  /**
	 * ******************* SetInputImageRegion *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::SetInputImageRegion( const InputImageRegionType _arg, unsigned int pos )
  {
    if ( this->m_InputImageRegionVector.size() < pos + 1 )
    {
      this->m_InputImageRegionVector.resize( pos + 1 );
      this->m_NumberOfInputImageRegions = pos + 1;
    }
    if ( pos == 0 )
    {
      this->m_InputImageRegion = _arg;
    }
    if ( this->m_InputImageRegionVector[ pos ] != _arg )
    {
      this->m_InputImageRegionVector[ pos ] = _arg;
      this->Modified();
    }

  } // SetInputImageRegion()


  /**
	 * ******************* GetInputImageRegion *******************
	 */
  
  template< class TInputImage >
    const typename ImageSamplerBase< TInputImage >::InputImageRegionType &
    ImageSamplerBase< TInputImage >
    ::GetInputImageRegion( unsigned int pos ) const
  {
    if ( this->m_InputImageRegionVector.size() < pos + 1 )
    {
      return this->m_DummyInputImageRegion;
    }
    return this->m_InputImageRegionVector[ pos ];

  } // end GetInputImageRegion()


  /**
	 * ******************* SetNumberOfInputImageRegions *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::SetNumberOfInputImageRegions( const unsigned int _arg )
  {
    if ( this->m_NumberOfInputImageRegions != _arg )
    {
      this->m_InputImageRegionVector.resize( _arg );
      this->m_NumberOfInputImageRegions = _arg;
      this->Modified();
    }

  } // end SetNumberOfInputImageRegions()


  /**
	 * ******************* GenerateInputRequestedRegion *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::GenerateInputRequestedRegion( void )
  {
    /** Check if input image was set. */
    if ( this->GetNumberOfInputs() == 0 )
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
    /** Set the Modified flag, such that on calling Update(), 
     * the GenerateData method is executed again. 
     * Return true to indicate that indeed new samples will be selected.
     * Inheriting subclasses may just return false and do nothing.
     */
    this->Modified();
    return true;

  } // end SelectNewSamplesOnUpdate


  /**
	 * ******************* IsInsideAllMasks *******************
	 */
  
  template< class TInputImage >
    bool
    ImageSamplerBase< TInputImage >
    ::IsInsideAllMasks( const InputImagePointType & point ) const
  {
    bool ret = true;
    for ( unsigned int i = 0; i < this->m_NumberOfMasks; ++i )
    {
      ret &= this->GetMask( i )->IsInside( point );
    }

    return ret;

  } // end IsInsideAllMasks()


  /**
	 * ******************* UpdateAllMasks *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::UpdateAllMasks( void )
  {
    /** If the masks are generated by a filter, then make sure they are updated. */
    for ( unsigned int i = 0; i < this->m_NumberOfMasks; ++i )
    {
      if ( this->GetMask( i )->GetSource() )
      {
        this->GetMask( i )->GetSource()->Update();
      }
    }

  } // end UpdateAllMasks()


  /**
	 * ******************* CheckInputImageRegions *******************
	 */
  
  template< class TInputImage >
    bool
    ImageSamplerBase< TInputImage >
    ::CheckInputImageRegions( void )
  {
    bool ret = true;
    for ( unsigned int i = 0; i < this->GetNumberOfInputImageRegions(); ++i )
    {
      ret &= this->GetInput( i )->GetLargestPossibleRegion().IsInside(
        this->GetInputImageRegion( i ) );
    }
    return ret;

  } // end CheckInputImageRegions()


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageSamplerBase< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "Mask: " << this->m_Mask.GetPointer() << std::endl;

  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __ImageSamplerBase_txx

