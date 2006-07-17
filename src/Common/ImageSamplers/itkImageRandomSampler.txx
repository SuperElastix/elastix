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
    RandomIteratorType randIter( inputImage, this->GetInputImageRegion() );
    randIter.GoToBegin();

    /** Setup an iterator over the output, which is of ImageSampleContainerType. */
    typename ImageSampleContainerType::Iterator iter;
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Fill the sample container. */
    if ( mask.IsNull() )
    {
      randIter.SetNumberOfSamples( this->GetNumberOfSamples() );
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
      /** No real meaning in this: */
      randIter.SetNumberOfSamples( this->GetNumberOfSamples() );
      /** Loop over the sample container. */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Loop untill a valid sample is found. */
        do 
        {
          /** Jump to a random position. */
          ++randIter;
          /** Get the index, and transform it to the physical coordinates. */
          InputImageIndexType index = randIter.GetIndex();
          inputImage->TransformIndexToPhysicalPoint( index,
            inputPoint );
        } while ( !mask->IsInside( inputPoint ) );
        /** Put the coordinates and the value in the sample. */
        (*iter).Value().m_ImageCoordinates = inputPoint;
        (*iter).Value().m_ImageValue = randIter.Get();
      } // end for loop
    } // end if mask

  } // end GenerateData


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageRandomSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf



} // end namespace itk

#endif // end #ifndef __ImageRandomSampler_txx

