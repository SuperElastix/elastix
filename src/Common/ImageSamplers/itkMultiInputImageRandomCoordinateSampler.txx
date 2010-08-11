/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __MultiInputImageRandomCoordinateSampler_txx
#define __MultiInputImageRandomCoordinateSampler_txx

#include "itkMultiInputImageRandomCoordinateSampler.h"
#include "vnl/vnl_inverse.h"
#include "itkConfigure.h"


namespace itk
{

  /**
   * ******************* Constructor ********************
   */

  template< class TInputImage >
    MultiInputImageRandomCoordinateSampler< TInputImage >
    ::MultiInputImageRandomCoordinateSampler()
  {
    /** Set the default interpolator. */
    typename DefaultInterpolatorType::Pointer bsplineInterpolator =
      DefaultInterpolatorType::New();
    bsplineInterpolator->SetSplineOrder( 3 );
    this->m_Interpolator = bsplineInterpolator;

    /** Setup the random generator. */
    this->m_RandomGenerator = RandomGeneratorType::New();

    this->m_UseRandomSampleRegion = false;
    this->m_SampleRegionSize.Fill( 1.0 );

  } // end Constructor()


  /**
   * ******************* GenerateData *******************
   */

  template< class TInputImage >
    void
    MultiInputImageRandomCoordinateSampler< TInputImage >
    ::GenerateData( void )
  {
    /** Check. */
    if ( !this->CheckInputImageRegions() )
    {
      itkExceptionMacro( << "ERROR: at least one of the InputImageRegions "
        << "is not a subregion of the LargestPossibleRegion" );
    }

    /** Get handles to the input image, output sample container, and mask. */
    InputImageConstPointer inputImage = this->GetInput();
    typename ImageSampleContainerType::Pointer sampleContainer = this->GetOutput();
    typename MaskType::ConstPointer mask = this->GetMask();
    typename InterpolatorType::Pointer interpolator = this->GetInterpolator();

    /** Set up the interpolator. */
    interpolator->SetInputImage( inputImage );

    /** Get the intersection of all sample regions. */
    InputImageContinuousIndexType smallestContIndex;
    InputImageContinuousIndexType largestContIndex;
    this->GenerateSampleRegion(
      smallestContIndex, largestContIndex );

    /** Reserve memory for the output. */
    sampleContainer->Reserve( this->GetNumberOfSamples() );

    /** Setup an iterator over the output, which is of ImageSampleContainerType. */
    typename ImageSampleContainerType::Iterator iter;
    typename ImageSampleContainerType::ConstIterator end = sampleContainer->End();

    InputImageContinuousIndexType sampleContIndex;
    /** Fill the sample container. */
    if ( mask.IsNull() )
    {
      /** Start looping over the sample container. */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container. */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;

        /** Generate a point in the input image region. */
        this->GenerateRandomCoordinate( smallestContIndex, largestContIndex, sampleContIndex );

        /** Convert to point */
        inputImage->TransformContinuousIndexToPhysicalPoint( sampleContIndex, samplePoint );

        /** Compute the value at the contindex. */
        sampleValue = static_cast<ImageSampleValueType>(
          this->m_Interpolator->EvaluateAtContinuousIndex( sampleContIndex ) );

      } // end for loop
    } // end if no mask
    else
    {
      /** Update all masks. */
      this->UpdateAllMasks();

      /** Set up some variable that are used to make sure we are not forever
       * walking around on this image, trying to look for valid samples.
       */
      unsigned long numberOfSamplesTried = 0;
      unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();

      /** Start looping over the sample container. */
      for ( iter = sampleContainer->Begin(); iter != end; ++iter )
      {
        /** Make a reference to the current sample in the container. */
        InputImagePointType & samplePoint = (*iter).Value().m_ImageCoordinates;
        ImageSampleValueType & sampleValue = (*iter).Value().m_ImageValue;

        /** Walk over the image until we find a valid point. */
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
            itkExceptionMacro( << "Could not find enough image samples within "
              << "reasonable time. Probably the mask is too small" );
          }

          /** Generate a point in the input image region. */
          this->GenerateRandomCoordinate( smallestContIndex, largestContIndex, sampleContIndex );
          inputImage->TransformContinuousIndexToPhysicalPoint( sampleContIndex, samplePoint );
        } while ( !this->IsInsideAllMasks( samplePoint ) );

        /** Compute the value at the contindex. */
        sampleValue = static_cast<ImageSampleValueType>(
          this->m_Interpolator->EvaluateAtContinuousIndex( sampleContIndex ) );

      } // end for loop
    } // end if mask

  } // end GenerateData()


  /**
   * ******************* GenerateSampleRegion *******************
   */

  template< class TInputImage >
    void
    MultiInputImageRandomCoordinateSampler< TInputImage >::
    GenerateSampleRegion(
      InputImageContinuousIndexType & smallestContIndex,
      InputImageContinuousIndexType & largestContIndex )
  {
    /** Get handles to the number of inputs and regions. */
    const unsigned int numberOfInputs = this->GetNumberOfInputs();
    const unsigned int numberOfRegions = this->GetNumberOfInputImageRegions();

    /** Check. */
    if ( numberOfRegions != numberOfInputs && numberOfRegions != 1 )
    {
      itkExceptionMacro( << "ERROR: The number of regions should be 1 or the number of inputs." );
    }

    typedef typename InputImageType::DirectionType DirectionType;
    DirectionType dir0 = this->GetInput( 0 )->GetDirection();
		typename DirectionType::InternalMatrixType dir0invtemp =
      vnl_inverse( dir0.GetVnlMatrix() );
		DirectionType dir0inv( dir0invtemp );
    for (unsigned int i = 1; i < numberOfInputs; ++i )
    {
      DirectionType diri = this->GetInput( i )->GetDirection();
      if ( diri != dir0 )
      {
        itkExceptionMacro( << "ERROR: All input images should have the same direction cosines matrix." );
      }
    }

    /** Initialize the smallest and largest point. */
    InputImagePointType smallestPoint;
    InputImagePointType largestPoint;
    smallestPoint.Fill( NumericTraits<InputImagePointValueType>::NonpositiveMin() );
    largestPoint.Fill( NumericTraits<InputImagePointValueType>::max() );

    /** Determine the intersection of all regions, assuming identical direction cosines,
     * but possibly different origin/spacing.
     * \todo: test this really carefully!
     */
    InputImageSizeType unitSize;
    unitSize.Fill( 1 );
    for ( unsigned int i = 0; i < numberOfRegions; ++i )
    {
      /** Get the outer indices. */
      InputImageIndexType smallestIndex
        = this->GetInputImageRegion( i ).GetIndex();
      InputImageIndexType largestIndex
        = smallestIndex + this->GetInputImageRegion( i ).GetSize() - unitSize;

      /** Convert to points */
      InputImagePointType smallestImagePoint;
      InputImagePointType largestImagePoint;
      this->GetInput( i )->TransformIndexToPhysicalPoint(
        smallestIndex, smallestImagePoint );
      this->GetInput( i )->TransformIndexToPhysicalPoint(
      largestIndex, largestImagePoint );

      /** apply inverse direction, so that next max-operation makes sense. */
#ifdef ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE
      smallestImagePoint = dir0inv * smallestImagePoint;
      largestImagePoint = dir0inv * largestImagePoint;
#endif

      /** Determine intersection. */
      for ( unsigned int j = 0; j < InputImageDimension; ++j )
      {
        /** Get the largest smallest point. */
        smallestPoint[ j ] = vnl_math_max( smallestPoint[ j ], smallestImagePoint[ j ] );

        /** Get the smallest largest point. */
        largestPoint[ j ] = vnl_math_min( largestPoint[ j ], largestImagePoint[ j ] );
      }
    }

    /** Convert to continuous index in input image 0. */
#ifdef ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE
    smallestPoint = dir0 * smallestPoint;
    largestPoint = dir0 * largestPoint;
#endif
    this->GetInput( 0 )->TransformPhysicalPointToContinuousIndex( smallestPoint, smallestContIndex );
    this->GetInput( 0 )->TransformPhysicalPointToContinuousIndex( largestPoint, largestContIndex );

    /** Support for localised mutual information. */
    if ( this->GetUseRandomSampleRegion() )
    {
      /** Convert sampleRegionSize to continuous index space */
      typedef typename InputImageContinuousIndexType::VectorType   CIndexVectorType;
      CIndexVectorType sampleRegionSize;
      for (unsigned int i = 0; i < InputImageDimension; ++i)
      {
        sampleRegionSize[i] = this->GetSampleRegionSize()[i] /
          this->GetInput()->GetSpacing()[i];
      }
      InputImageContinuousIndexType maxSmallestContIndex = largestContIndex;
      maxSmallestContIndex -= sampleRegionSize;
      this->GenerateRandomCoordinate( smallestContIndex, maxSmallestContIndex, smallestContIndex );
      largestContIndex = smallestContIndex;
      largestContIndex += sampleRegionSize;
    }

  } // end GenerateSampleRegion()


  /**
   * ******************* GenerateRandomCoordinate *******************
   */

  template< class TInputImage >
    void
    MultiInputImageRandomCoordinateSampler< TInputImage >::
    GenerateRandomCoordinate(
      const InputImageContinuousIndexType & smallestContIndex,
      const InputImageContinuousIndexType & largestContIndex,
      InputImageContinuousIndexType &       randomContIndex)
  {
    for ( unsigned int i = 0; i < InputImageDimension; ++i)
    {
      randomContIndex[ i ] = static_cast<InputImagePointValueType>(
        this->m_RandomGenerator->GetUniformVariate(
        smallestContIndex[ i ], largestContIndex[ i ] ) );
    }
  } // end GenerateRandomCoordinate()


  /**
   * ******************* PrintSelf *******************
   */

  template< class TInputImage >
    void
    MultiInputImageRandomCoordinateSampler< TInputImage >
    ::PrintSelf( std::ostream& os, Indent indent ) const
  {
    Superclass::PrintSelf( os, indent );

    os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;
    os << indent << "RandomGenerator: " << this->m_RandomGenerator.GetPointer() << std::endl;

  } // end PrintSelf


} // end namespace itk

#endif // end #ifndef __MultiInputImageRandomCoordinateSampler_txx

