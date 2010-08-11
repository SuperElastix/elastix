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
    /** Get handles to the input image, output sample container, and mask. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
    typename MaskType::ConstPointer mask = this->GetMask();

    /** Reserve memory for the output. */
    sampleContainer->Reserve( this->GetNumberOfSamples() );

    /** Setup a random iterator over the input image. */
    typedef ImageRandomConstIteratorWithIndex< InputImageType > RandomIteratorType;
    RandomIteratorType randIter( inputImage, this->GetCroppedInputImageRegion() );
    randIter.GoToBegin();

    /** Setup an iterator over the output, which is of ImageSampleContainerType. */
    typename ImageSampleContainerType::Iterator iter;
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Fill the sample container. */
    if ( mask.IsNull() )
    {
      /** number of samples + 1, because of the initial ++randIter. */
      randIter.SetNumberOfSamples( this->GetNumberOfSamples()+1 );
      /** Advance one, in order to generate the same sequence as when using a mask */
      ++randIter;
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Get the index, transform it to the physical coordinates and put it in the sample. */
        InputImageIndexType index = randIter.GetIndex();
        inputImage->TransformIndexToPhysicalPoint( index,
          (*iter).Value().m_ImageCoordinates );
        /** Get the value and put it in the sample. */
        (*iter).Value().m_ImageValue = randIter.Get();
        /** Jump to a random position. */
        ++randIter;

      } // end for loop
    } // end if no mask
    else
    {
      if ( mask->GetSource() )
      {
        mask->GetSource()->Update();
      }
      InputImagePointType inputPoint;
      bool insideMask = false;
      /** Make sure we are not eternally trying to find samples: */
      randIter.SetNumberOfSamples( 10 * this->GetNumberOfSamples() );
      /** Loop over the sample container. */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Loop until a valid sample is found. */
        do
        {
          /** Jump to a random position. */
          ++randIter;
          /** Check if we are not trying eternally to find a valid point. */
          if ( randIter.IsAtEnd() )
          {
            /** Squeeze the sample container to the size that is still valid. */
            typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
            typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
            stlnow += iter.Index();
            sampleContainer->erase( stlnow, stlend );
            itkExceptionMacro( << "Could not find enough image samples within "
              << "reasonable time. Probably the mask is too small" );
          }
          /** Get the index, and transform it to the physical coordinates. */
          InputImageIndexType index = randIter.GetIndex();
          inputImage->TransformIndexToPhysicalPoint( index, inputPoint );
          /** Check if it's inside the mask. */
          insideMask = mask->IsInside( inputPoint );
        } while ( !insideMask );

        /** Put the coordinates and the value in the sample. */
        (*iter).Value().m_ImageCoordinates = inputPoint;
        (*iter).Value().m_ImageValue = randIter.Get();

      } // end for loop

      /** Extra random sample to make sure the same sequence is generated
       * with and without mask. */
      ++randIter;

    } // end if mask


  } // end GenerateData()


} // end namespace itk

#endif // end #ifndef __ImageRandomSampler_txx

