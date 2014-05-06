/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxViolaWellsMutualInformationMetric_HXX__
#define __elxViolaWellsMutualInformationMetric_HXX__

#include "elxViolaWellsMutualInformationMetric.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
ViolaWellsMutualInformationMetric< TElastix >
::ViolaWellsMutualInformationMetric()
{}  // end Constructor

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
ViolaWellsMutualInformationMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of ViolaWellsMutualInformationMetric metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
ViolaWellsMutualInformationMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Set the number of histogram bins and spatial samples. */
  unsigned int numberOfSpatialSamples = 10000;

  /** Set the intensity standard deviation of the fixed
   * and moving images. This defines the kernel bandwidth
   * used in the joint probability distribution calculation.
   * Default value is 0.4 which works well for image intensities
   * normalized to a mean of 0 and standard deviation of 1.0.
   * Value is clamped to be always greater than zero.
   */
  double fixedImageStandardDeviation  = 0.4;
  double movingImageStandardDeviation = 0.4;
  /** \todo calculate them??? */

  /** Read the parameters from the ParameterFile. */
  this->m_Configuration->ReadParameter( numberOfSpatialSamples,
    "NumberOfSpatialSamples", this->GetComponentLabel(), level, 0 );
  this->m_Configuration->ReadParameter( fixedImageStandardDeviation,
    "FixedImageStandardDeviation", this->GetComponentLabel(), level, 0 );
  this->m_Configuration->ReadParameter( movingImageStandardDeviation,
    "MovingImageStandardDeviation", this->GetComponentLabel(), level, 0 );

  /** Set them. */
  this->SetNumberOfSpatialSamples( numberOfSpatialSamples );
  this->SetFixedImageStandardDeviation( fixedImageStandardDeviation );
  this->SetMovingImageStandardDeviation( movingImageStandardDeviation );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxViolaWellsMutualInformationMetric_HXX__
