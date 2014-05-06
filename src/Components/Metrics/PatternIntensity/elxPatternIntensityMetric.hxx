/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxPatternIntensityMetric_HXX__
#define __elxPatternIntensityMetric_HXX__

#include "elxPatternIntensityMetric.h"
#include "itkTimeProbe.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
PatternIntensityMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of PatternIntensity metric took: "
    << static_cast< long >( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
PatternIntensityMetric< TElastix >
::BeforeRegistration( void )
{
  if( this->m_Elastix->GetFixedImage()->GetImageDimension() != 3 )
  {
    itkExceptionMacro( << "FixedImage must be 3D" );
  }
  if( this->m_Elastix->GetFixedImage()->GetImageDimension() == 3 )
  {
    if( this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize()[ 2 ] != 1 )
    {
      itkExceptionMacro( << "Metric can only be used for 2D-3D registration. FixedImageSize[2] must be 1" );
    }
  }

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
PatternIntensityMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level.*/
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

  /** Set noise constant, */
  double sigma = 100;
  this->m_Configuration->ReadParameter( sigma,
    "Sigma", this->GetComponentLabel(), level, 0 );
  this->SetNoiseConstant( sigma * sigma );

  /** Set optimization of normalization factor. */
  bool optimizenormalizationfactor = false;
  this->m_Configuration->ReadParameter( optimizenormalizationfactor,
    "OptimizeNormalizationFactor", this->GetComponentLabel(), level, 0 );
  this->SetOptimizeNormalizationFactor( optimizenormalizationfactor );

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  bool                            usescales = true;
  for( unsigned int i = 0; i < MovingImageDimension; ++i )
  {
    usescales &= this->GetConfiguration()->ReadParameter(
      movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
      this->GetComponentLabel(), i, -1, false );
  }
  if( usescales )
  {
    this->SetUseMovingImageDerivativeScales( true );
    this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
    elxout << "Multiplying moving image derivatives by: "
           << movingImageDerivativeScales << std::endl;
  }

  typedef typename elastix::OptimizerBase< TElastix >::ITKBaseType::ScalesType ScalesType;
  ScalesType scales = this->m_Elastix->GetElxOptimizerBase()->GetAsITKBaseType()->GetScales();
  this->SetScales( scales );

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxPatternIntensityMetric_HXX__
