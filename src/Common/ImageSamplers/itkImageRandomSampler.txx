/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __ImageRandomSampler_txx
#define __ImageRandomSampler_txx

#include "itkImageRandomSampler.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkImageRandomConstIteratorWithIndex.h"

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSampler< TInputImage >
::GenerateData( void )
{
  /** Get a handle to the mask. If there was no mask supplied we exercise a multi-threaded version. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if( mask.IsNull() && this->m_UseMultiThread )
  {
    /** Calls ThreadedGenerateData(). */
    return Superclass::GenerateData();
  }

  /** Get handles to the input image, output sample container. */
  InputImageConstPointer inputImage = this->GetInput();
  typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();

  /** Reserve memory for the output. */
  sampleContainer->Reserve( this->GetNumberOfSamples() );

  /** Setup a random iterator over the input image. */
  typedef ImageRandomConstIteratorWithIndex< InputImageType > RandomIteratorType;
  RandomIteratorType randIter( inputImage, this->GetCroppedInputImageRegion() );
  randIter.GoToBegin();

  /** Setup an iterator over the output, which is of ImageSampleContainerType. */
  typename ImageSampleContainerType::Iterator iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

  if( mask.IsNull() )
  {
    /** number of samples + 1, because of the initial ++randIter. */
    randIter.SetNumberOfSamples( this->GetNumberOfSamples() + 1 );
    /** Advance one, in order to generate the same sequence as when using a mask */
    ++randIter;
    for( iter = sampleContainer->Begin(); iter != end; ++iter )
    {
      /** Get the index, transform it to the physical coordinates and put it in the sample. */
      InputImageIndexType index = randIter.GetIndex();
      inputImage->TransformIndexToPhysicalPoint( index,
        ( *iter ).Value().m_ImageCoordinates );
      /** Get the value and put it in the sample. */
      ( *iter ).Value().m_ImageValue = randIter.Get();
      /** Jump to a random position. */
      ++randIter;

    } // end for loop
  }   // end if no mask
  else
  {
    /** Update the mask. */
    if( mask->GetSource() )
    {
      mask->GetSource()->Update();
    }

    /** Make sure we are not eternally trying to find samples: */
    randIter.SetNumberOfSamples( 10 * this->GetNumberOfSamples() );

    /** Loop over the sample container. */
    InputImagePointType inputPoint;
    bool                insideMask = false;
    for( iter = sampleContainer->Begin(); iter != end; ++iter )
    {
      /** Loop until a valid sample is found. */
      do
      {
        /** Jump to a random position. */
        ++randIter;
        /** Check if we are not trying eternally to find a valid point. */
        if( randIter.IsAtEnd() )
        {
          /** Squeeze the sample container to the size that is still valid. */
          typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
          typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
          stlnow                                            += iter.Index();
          sampleContainer->erase( stlnow, stlend );
          itkExceptionMacro( << "Could not find enough image samples within "
                             << "reasonable time. Probably the mask is too small" );
        }
        /** Get the index, and transform it to the physical coordinates. */
        InputImageIndexType index = randIter.GetIndex();
        inputImage->TransformIndexToPhysicalPoint( index, inputPoint );
        /** Check if it's inside the mask. */
        insideMask = mask->IsInside( inputPoint );
      }
      while( !insideMask );

      /** Put the coordinates and the value in the sample. */
      ( *iter ).Value().m_ImageCoordinates = inputPoint;
      ( *iter ).Value().m_ImageValue       = randIter.Get();

    } // end for loop

    /** Extra random sample to make sure the same sequence is generated
     * with and without mask.
     */
    ++randIter;
  }

} // end GenerateData()


/**
 * ******************* ThreadedGenerateData *******************
 */

template< class TInputImage >
void
ImageRandomSampler< TInputImage >
::ThreadedGenerateData( const InputImageRegionType &, ThreadIdType threadId )
{
  /** Sanity check. */
  typename MaskType::ConstPointer mask = this->GetMask();
  if( mask.IsNotNull() )
  {
    itkExceptionMacro( << "ERROR: do not call this function when a mask is supplied." );
  }

  /** Get handle to the input image. */
  InputImageConstPointer inputImage = this->GetInput();

  /** Figure out which samples to process. */
  unsigned long chunkSize   = this->GetNumberOfSamples() / this->GetNumberOfThreads();
  unsigned long sampleStart = threadId * chunkSize;
  if( threadId == this->GetNumberOfThreads() - 1 )
  {
    chunkSize = this->GetNumberOfSamples()
      - ( ( this->GetNumberOfThreads() - 1 ) * chunkSize );
  }

  /** Get a reference to the output and reserve memory for it. */
  ImageSampleContainerPointer & sampleContainerThisThread
    = this->m_ThreaderSampleContainer[ threadId ];
  sampleContainerThisThread->Reserve( chunkSize );

  /** Setup an iterator over the sampleContainerThisThread. */
  typename ImageSampleContainerType::Iterator iter;
  typename ImageSampleContainerType::ConstIterator end = sampleContainerThisThread->End();

  /** Fill the local sample container. */
  unsigned long       sampleId    = sampleStart;
  InputImageSizeType  regionSize  = this->GetCroppedInputImageRegion().GetSize();
  InputImageIndexType regionIndex = this->GetCroppedInputImageRegion().GetIndex();
  for( iter = sampleContainerThisThread->Begin(); iter != end; ++iter, sampleId++ )
  {
    unsigned long randomPosition = static_cast< unsigned long >( this->m_RandomNumberList[ sampleId ] );

    /** Translate randomPosition to an index, copied from ImageRandomConstIteratorWithIndex. */
    unsigned long       residual;
    InputImageIndexType positionIndex;
    for( unsigned int dim = 0; dim < InputImageDimension; dim++ )
    {
      const unsigned long sizeInThisDimension = regionSize[ dim ];
      residual             = randomPosition % sizeInThisDimension;
      positionIndex[ dim ] = residual + regionIndex[ dim ];
      randomPosition      -= residual;
      randomPosition      /= sizeInThisDimension;
    }

    /** Transform index to the physical coordinates and put it in the sample. */
    inputImage->TransformIndexToPhysicalPoint( positionIndex,
      ( *iter ).Value().m_ImageCoordinates );

    /** Get the value and put it in the sample. */
    ( *iter ).Value().m_ImageValue = static_cast< ImageSampleValueType >( inputImage->GetPixel( positionIndex ) );

  } // end for loop

} // end ThreadedGenerateData()


} // end namespace itk

#endif // end #ifndef __ImageRandomSampler_txx
