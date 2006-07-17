#ifndef __ImageRandomCoordinateSampler_txx
#define __ImageRandomCoordinateSampler_txx

#include "itkImageRandomCoordinateSampler.h"


namespace itk
{

  /**
	 * ******************* Constructor ********************
	 */
  
  template< class TInputImage >
    ImageRandomCoordinateSampler< TInputImage > ::
    ImageRandomCoordinateSampler()
  {
    this->m_NumberOfSamples = 100;

    /** Set default interpolator */
    typename DefaultInterpolatorType::Pointer bsplineInterpolator =
      DefaultInterpolatorType::New();
    bsplineInterpolator->SetSplineOrder(3);
    this->m_Interpolator = bsplineInterpolator;

    /** Setup random generator */
    this->m_RandomGenerator = RandomGeneratorType::New();
    //this->m_RandomGenerator->Initialize();

  } // end constructor 


  /**
	 * ******************* GenerateData *******************
	 */
  
  template< class TInputImage >
    void
    ImageRandomCoordinateSampler< TInputImage >
    ::GenerateData( void )
  {
    /** Get handles to the input image, output sample container, and mask. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
    typename MaskType::ConstPointer mask = this->GetMask();
    typename InterpolatorType::Pointer interpolator = this->GetInterpolator();

    /** Set up the interpolator */
    interpolator->SetInputImage( inputImage );

    /** Convert inputImageRegion to bounding box in physical space */
    InputImageIndexType smallestIndex = this->GetInputImageRegion().GetIndex();
    InputImageIndexType largestIndex = smallestIndex + this->GetInputImageRegion().GetSize();
    InputImagePointType smallestPoint;
    InputImagePointType largestPoint;
    inputImage->TransformIndexToPhysicalPoint(
      smallestIndex, smallestPoint);
    inputImage->TransformIndexToPhysicalPoint(
      largestIndex, largestPoint);
    
    /** Reserve memory for the output. */
    sampleContainer->Reserve( this->GetNumberOfSamples() );
   
    /** Setup an iterator over the output, which is of ImageSampleContainerType. */
    typename ImageSampleContainerType::Iterator iter;
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Fill the sample container. */
    if ( mask.IsNull() )
    {
      /** Start looping over the sample container */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;
        /** Walk over the image until we find a valid point */
        do 
        {
          /** Generate a point in the input image region */
          this->GenerateRandomCoordinate( smallestPoint, largestPoint, samplePoint );
        } while ( !interpolator->IsInsideBuffer( samplePoint ) );
        /** Compute the value at the point */
        sampleValue = static_cast<ImageSampleValueType>(
          this->m_Interpolator->Evaluate( samplePoint ) );
      } // end for loop
    } // end if no mask
    else
    {
      /** Update the mask */
      if ( mask->GetSource() )
      {
        mask->GetSource()->Update();
      }
      /** Set up some variable that are used to make sure we are not forever
       * walking around on this image, trying to look for valid samples */
      unsigned long numberOfSamplesTried = 0;
      unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();
      /** Start looping over the sample container */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;
        /** Walk over the image until we find a valid point */
        do 
        {
          /** Check if we are not trying eternally to find a valid point. */
          ++numberOfSamplesTried;
          if ( numberOfSamplesTried > maximumNumberOfSamplesToTry )
          {
            /** Squeeze the sample container to the size that is still valid */
            ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
            ImageSampleContainerType::iterator stlend = sampleContainer->end();
            stlnow += iter.Index();
            sampleContainer->erase( stlnow, stlend);
            /** Throw an error */
            itkExceptionMacro( << "Could not find enough image samples within reasonable time. Probably the mask is too small" );
          }
          /** Generate a point in the input image region */
          this->GenerateRandomCoordinate( smallestPoint, largestPoint, samplePoint );
        } while ( !interpolator->IsInsideBuffer( samplePoint ) || 
                  !mask->IsInside( samplePoint ) );
        /** Compute the value at the point */
        sampleValue = static_cast<ImageSampleValueType>( 
          this->m_Interpolator->Evaluate( samplePoint ) );
      } // end for loop
    } // end if mask
   
  } // end GenerateData


  /**
	 * ******************* GenerateRandomCoordinate *******************
	 */
  
  template< class TInputImage >
    void
    ImageRandomCoordinateSampler< TInputImage >::
    GenerateRandomCoordinate(
      const InputImagePointType & smallestPoint,
      const InputImagePointType & largestPoint,
      InputImagePointType &       randomPoint)
  {
    for ( unsigned int i = 0; i < InputImageDimension; ++i)
    {
      randomPoint[i] = static_cast<InputImagePointValueType>( 
        this->m_RandomGenerator->GetUniformVariate(
        smallestPoint[i], largestPoint[i] ) );
    }
  } // end GenerateRandomCoordinate   


  /**
	 * ******************* PrintSelf *******************
	 */
  
  template< class TInputImage >
    void
    ImageRandomCoordinateSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );
  } // end PrintSelf



} // end namespace itk

#endif // end #ifndef __ImageRandomCoordinateSampler_txx

