/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef _itkErodeMaskImageFilter_hxx
#define _itkErodeMaskImageFilter_hxx

#include "itkErodeMaskImageFilter.h"
#include "itkParabolicErodeImageFilter.h"
//#include "itkThresholdImageFilter.h"

namespace itk
{

/**
 * ************* Constructor *******************
 */

template< class TImage >
ErodeMaskImageFilter< TImage >
::ErodeMaskImageFilter()
{
  this->m_IsMovingMask    = false;
  this->m_ResolutionLevel = 0;

  ScheduleType defaultSchedule( 1, InputImageDimension );
  defaultSchedule.Fill( NumericTraits< unsigned int >::One );
  this->m_Schedule = defaultSchedule;

} // end Constructor


/**
 * ************* GenerateData *******************
 */

template< class TImage >
void
ErodeMaskImageFilter< TImage >
::GenerateData( void )
{
  /** Typedefs. */
  //typedef itk::ThresholdImageFilter<InputImageType> ThresholdFilterType;
  typedef itk::ParabolicErodeImageFilter<
    InputImageType, OutputImageType >               ErodeFilterType;
  typedef typename ErodeFilterType::RadiusType     RadiusType;
  typedef typename ErodeFilterType::ScalarRealType ScalarRealType;

  /** Get the correct radius. */
  RadiusType     radiusarray;
  ScalarRealType radius   = 0.0;
  ScalarRealType schedule = 0.0;
  for( unsigned int i = 0; i < InputImageDimension; ++i )
  {
    schedule = static_cast< ScalarRealType >(
      this->GetSchedule()[ this->GetResolutionLevel() ][ i ] );
    if( !this->GetIsMovingMask() )
    {
      radius = schedule + 1.0;
    }
    else
    {
      radius = 2.0 * schedule + 1.0;
    }
    // Very specific computation for the parabolic erosion filter:
    radius = radius * radius / 2.0 + 1.0;

    radiusarray.SetElement( i, radius );
  }

  /** Threshold the data first. Every voxel with intensity >= 1 is used.
  // Not needed since IsInside of a mask checks for != 0.
  typename ThresholdFilterType::Pointer threshold = ThresholdFilterType::New();
  threshold->ThresholdAbove(  itk::NumericTraits<InputPixelType>::One );
  threshold->SetOutsideValue( itk::NumericTraits<InputPixelType>::One );
  threshold->SetInput( this->GetInput() ); */

  /** Create and run the erosion filter. */
  typename ErodeFilterType::Pointer erosion = ErodeFilterType::New();
  erosion->SetUseImageSpacing( false );
  erosion->SetScale( radiusarray );
  //erosion->SetInput( threshold->GetOutput() );
  erosion->SetInput( this->GetInput() );
  erosion->Update();

  /** Graft the output of the mini-pipeline back onto the filter's output.
   * this copies back the region ivars and meta-data.
   */
  this->GraftOutput( erosion->GetOutput() );

} // end GenerateData()


} // end namespace itk

#endif
