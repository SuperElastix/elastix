/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __ImageFullSampler_txx
#define __ImageFullSampler_txx

#include "itkImageFullSampler.h"

#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template< class TInputImage >
void
ImageFullSampler< TInputImage >
::GenerateData( void )
{
  /** If desired we exercise a multi-threaded version. */
  if( this->m_UseMultiThread )
  {
    /** Calls ThreadedGenerateData(). */
    return Superclass::GenerateData();
  }

  /** Get handles to the input image, output sample container, and the mask. */
  InputImageConstPointer inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
  typename MaskType::ConstPointer mask                       = this->GetMask();

  /** Clear the container. */
  sampleContainer->Initialize();

  /** Set up a region iterator within the user specified image region. */
  typedef ImageRegionConstIteratorWithIndex< InputImageType > InputImageIterator;
  InputImageIterator iter( inputImage, this->GetCroppedInputImageRegion() );

  /** Fill the sample container. */
  if( mask.IsNull() )
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleContainer->Reserve( this->GetCroppedInputImageRegion()
        .GetNumberOfPixels() );
    }
    catch( std::exception & excp )
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      const char * message2 = message.c_str();
      itkExceptionMacro( << message2 );
    }
    catch( ... )
    {
      itkExceptionMacro( << "ERROR: failed to allocate memory for the sample container." );
    }

    /** Simply loop over the image and store all samples in the container. */
    ImageSampleType tempSample;
    unsigned long   ind = 0;
    for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter, ++ind )
    {
      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint( index,
        tempSample.m_ImageCoordinates );

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container */
      sampleContainer->SetElement( ind, tempSample );

    } // end for
  }   // end if no mask
  else
  {
    if( mask->GetSource() )
    {
      mask->GetSource()->Update();
    }

    /** Loop over the image and check if the points falls within the mask. */
    ImageSampleType tempSample;
    for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter )
    {
      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint( index,
        tempSample.m_ImageCoordinates );

      if( mask->IsInside( tempSample.m_ImageCoordinates ) )
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /** Store in container. */
        sampleContainer->push_back( tempSample );

      } // end if
    }   // end for
  }     // end else (if mask exists)

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageFullSampler< TInputImage >
::ThreadedGenerateData( const InputImageRegionType & inputRegionForThread,
  ThreadIdType threadId )
{
  /** Get handles to the input image, mask and the output. */
  InputImageConstPointer inputImage = this->GetInput();
  typename MaskType::ConstPointer mask = this->GetMask();
  ImageSampleContainerPointer & sampleContainerThisThread // & ???
    = this->m_ThreaderSampleContainer[ threadId ];

  /** Set up a region iterator within the user specified image region. */
  typedef ImageRegionConstIteratorWithIndex< InputImageType > InputImageIterator;
  //InputImageIterator iter( inputImage, this->GetCroppedInputImageRegion() );
  InputImageIterator iter( inputImage, inputRegionForThread );

  /** Fill the sample container. */
  const unsigned long chunkSize = inputRegionForThread.GetNumberOfPixels();
  if( mask.IsNull() )
  {
    /** Try to reserve memory. If no mask is used this can raise std
     * exceptions when the input image is large.
     */
    try
    {
      sampleContainerThisThread->Reserve( chunkSize );
    }
    catch( std::exception & excp )
    {
      std::string message = "std: ";
      message += excp.what();
      message += "\nERROR: failed to allocate memory for the sample container.";
      const char * message2 = message.c_str();
      itkExceptionMacro( << message2 );
    }
    catch( ... )
    {
      itkExceptionMacro( << "ERROR: failed to allocate memory for the sample container." );
    }

    /** Simply loop over the image and store all samples in the container. */
    ImageSampleType tempSample;
    unsigned long   ind = 0;
    for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter, ++ind )
    {
      /** Get sampled index */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point */
      inputImage->TransformIndexToPhysicalPoint( index,
        tempSample.m_ImageCoordinates );

      /** Get sampled image value */
      tempSample.m_ImageValue = iter.Get();

      /** Store in container. */
      sampleContainerThisThread->SetElement( ind, tempSample );

    } // end for
  }   // end if no mask
  else
  {
    if( mask->GetSource() )
    {
      mask->GetSource()->Update();
    }

    /** Loop over the image and check if the points falls within the mask. */
    ImageSampleType tempSample;
    for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter )
    {
      /** Get sampled index. */
      InputImageIndexType index = iter.GetIndex();

      /** Translate index to point. */
      inputImage->TransformIndexToPhysicalPoint( index,
        tempSample.m_ImageCoordinates );

      if( mask->IsInside( tempSample.m_ImageCoordinates ) )
      {
        /** Get sampled image value. */
        tempSample.m_ImageValue = iter.Get();

        /**  Store in container. */
        sampleContainerThisThread->push_back( tempSample );

      } // end if
    }   // end for
  }     // end else (if mask exists)

} // end ThreadedGenerateData()


/**
 * ******************* PrintSelf *******************
 */

template< class TInputImage >
void
ImageFullSampler< TInputImage >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __ImageFullSampler_txx
