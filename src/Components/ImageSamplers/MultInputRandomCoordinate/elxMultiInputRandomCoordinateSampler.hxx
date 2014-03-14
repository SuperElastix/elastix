/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMultiInputRandomCoordinateSampler_hxx
#define __elxMultiInputRandomCoordinateSampler_hxx

#include "elxMultiInputRandomCoordinateSampler.h"

namespace elastix
{

/**
* ******************* BeforeEachResolution ******************
*/

template< class TElastix >
void
MultiInputRandomCoordinateSampler< TElastix >
::BeforeEachResolution( void )
{
  const unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Set the NumberOfSpatialSamples. */
  unsigned long numberOfSpatialSamples = 5000;
  this->GetConfiguration()->ReadParameter( numberOfSpatialSamples,
    "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfSamples( numberOfSpatialSamples );

  /** Set up the fixed image interpolator and set the SplineOrder, default value = 1. */
  typename DefaultInterpolatorType::Pointer fixedImageInterpolator
    = DefaultInterpolatorType::New();
  unsigned int splineOrder = 1;
  this->GetConfiguration()->ReadParameter( splineOrder,
    "FixedImageBSplineInterpolationOrder", this->GetComponentLabel(), level, 0 );
  fixedImageInterpolator->SetSplineOrder( splineOrder );
  this->SetInterpolator( fixedImageInterpolator );

  /** Set the UseRandomSampleRegion bool. */
  bool useRandomSampleRegion = false;
  this->GetConfiguration()->ReadParameter( useRandomSampleRegion,
    "UseRandomSampleRegion", this->GetComponentLabel(), level, 0 );
  this->SetUseRandomSampleRegion( useRandomSampleRegion );

  /** Set the SampleRegionSize. */
  if( useRandomSampleRegion )
  {
    InputImageSpacingType sampleRegionSize;
    InputImageSpacingType fixedImageSpacing
      = this->GetElastix()->GetFixedImage()->GetSpacing();
    InputImageSizeType fixedImageSize
      = this->GetElastix()->GetFixedImage()->GetLargestPossibleRegion().GetSize();

    /** Estimate default:
     * sampleRegionSize[i] = min ( fixedImageSizeInMM[i], max_i ( fixedImageSizeInMM[i]/3 ) )
     */
    double maxthird = 0.0;
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      sampleRegionSize[ i ] = ( fixedImageSize[ i ] - 1 ) * fixedImageSpacing[ i ];
      maxthird              = vnl_math_max( maxthird, sampleRegionSize[ i ] / 3.0 );
    }
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      sampleRegionSize[ i ] = vnl_math_min( maxthird, sampleRegionSize[ i ] );
    }

    /** Read user's choice. */
    for( unsigned int i = 0; i < InputImageDimension; ++i )
    {
      this->GetConfiguration()->ReadParameter(
        sampleRegionSize[ i ], "SampleRegionSize",
        this->GetComponentLabel(), level * InputImageDimension + i, 0 );
    }
    this->SetSampleRegionSize( sampleRegionSize );
  }

}   // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxMultiInputRandomCoordinateSampler_hxx
