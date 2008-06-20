/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
    /** Set default interpolator */
    typename DefaultInterpolatorType::Pointer bsplineInterpolator =
      DefaultInterpolatorType::New();
    bsplineInterpolator->SetSplineOrder( 3 );
    this->m_Interpolator = bsplineInterpolator;

    /** Setup random generator */
    this->m_RandomGenerator = RandomGeneratorType::New();
    //this->m_RandomGenerator->Initialize();

    this->m_UseRandomSampleRegion = false;
    this->m_SampleRegionSize.Fill( 1.0 );

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

    /** Set up the interpolator. */
    interpolator->SetInputImage( inputImage );

    /** Convert inputImageRegion to bounding box in physical space. */
    InputImageSizeType unitSize;
    unitSize.Fill( 1 );
    InputImageIndexType smallestIndex
      = this->GetCroppedInputImageRegion().GetIndex();
    InputImageIndexType largestIndex
      = smallestIndex + this->GetCroppedInputImageRegion().GetSize() - unitSize;
    InputImagePointType smallestImagePoint;
    InputImagePointType largestImagePoint;
    inputImage->TransformIndexToPhysicalPoint(
      smallestIndex, smallestImagePoint );
    inputImage->TransformIndexToPhysicalPoint(
      largestIndex, largestImagePoint );
    InputImagePointType smallestPoint;
    InputImagePointType largestPoint;
    this->GenerateSampleRegion( smallestImagePoint, largestImagePoint, 
      smallestPoint, largestPoint );
    
    /** Reserve memory for the output. */
    sampleContainer->Reserve( this->GetNumberOfSamples() );
   
    /** Setup an iterator over the output, which is of ImageSampleContainerType. */
    typename ImageSampleContainerType::Iterator iter;
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    /** Fill the sample container. */
    if ( mask.IsNull() )
    {
      /** Start looping over the sample container. */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container. */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;

        /** Walk over the image until we find a valid point. */
        do 
        {
          /** Generate a point in the input image region. */
          this->GenerateRandomCoordinate( smallestPoint, largestPoint, samplePoint );
        } while ( !interpolator->IsInsideBuffer( samplePoint ) );

        /** Compute the value at the point. */
        sampleValue = static_cast<ImageSampleValueType>(
          this->m_Interpolator->Evaluate( samplePoint ) );

      } // end for loop
    } // end if no mask
    else
    {
      /** Update the mask. */
      if ( mask->GetSource() )
      {
        mask->GetSource()->Update();
      }
      /** Set up some variable that are used to make sure we are not forever
       * walking around on this image, trying to look for valid samples. */
      unsigned long numberOfSamplesTried = 0;
      unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();

      /** Start looping over the sample container */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container. */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;

        /** Walk over the image until we find a valid point */
        do 
        {
          /** Check if we are not trying eternally to find a valid point. */
          ++numberOfSamplesTried;
          if ( numberOfSamplesTried > maximumNumberOfSamplesToTry )
          {
            /** Squeeze the sample container to the size that is still valid. */
            typename ImageSampleContainerType::iterator stlnow = sampleContainer->begin();
            typename ImageSampleContainerType::iterator stlend = sampleContainer->end();
            stlnow += iter.Index();
            sampleContainer->erase( stlnow, stlend);
            /** Throw an error. */
            itkExceptionMacro( << "Could not find enough image samples within reasonable time. Probably the mask is too small" );
          }

          /** Generate a point in the input image region. */
          this->GenerateRandomCoordinate( smallestPoint, largestPoint, samplePoint );

        } while ( !interpolator->IsInsideBuffer( samplePoint ) || 
                  !mask->IsInside( samplePoint ) );

        /** Compute the value at the point. */
        sampleValue = static_cast<ImageSampleValueType>( 
          this->m_Interpolator->Evaluate( samplePoint ) );

      } // end for loop
    } // end if mask
   
  } // end GenerateData()


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
      randomPoint[ i ] = static_cast<InputImagePointValueType>( 
        this->m_RandomGenerator->GetUniformVariate(
        smallestPoint[ i ], largestPoint[ i ] ) );
    }
  } // end GenerateRandomCoordinate()


  /**
   * ******************* GenerateSampleRegion *******************
   */
  
  template< class TInputImage >
    void
    ImageRandomCoordinateSampler< TInputImage >::
    GenerateSampleRegion(
      const InputImagePointType & smallestImagePoint,
      const InputImagePointType & largestImagePoint,
      InputImagePointType &       smallestPoint,
      InputImagePointType &       largestPoint )
  {
    if ( !this->GetUseRandomSampleRegion() )
    {
      smallestPoint = smallestImagePoint;
      largestPoint = largestImagePoint;
      return;
    }
    InputImagePointType maxSmallestPoint = largestImagePoint - this->GetSampleRegionSize();
    this->GenerateRandomCoordinate( smallestImagePoint, maxSmallestPoint, smallestPoint );
    largestPoint = smallestPoint + this->GetSampleRegionSize();
    
  } // end GenerateRandomCoordinate()


  /**
   * ******************* PrintSelf *******************
   */
  
  template< class TInputImage >
    void
    ImageRandomCoordinateSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;
    os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

  } // end PrintSelf()


} // end namespace itk

#endif // end #ifndef __ImageRandomCoordinateSampler_txx

