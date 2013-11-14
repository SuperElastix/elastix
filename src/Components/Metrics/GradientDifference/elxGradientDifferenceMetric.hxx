/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxGradientDifferenceMetric_HXX__
#define __elxGradientDifferenceMetric_HXX__

#include "elxGradientDifferenceMetric.h"


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void GradientDifferenceMetric<TElastix>
::Initialize( void ) throw (itk::ExceptionObject)
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of GradientDifference metric took: "
    << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void GradientDifferenceMetric<TElastix>
::BeforeRegistration(void)
{

  if ( this->m_Elastix->GetFixedImage()->GetImageDimension() != 3 )
  {
    itkExceptionMacro( << "FixedImage must be 3D" );
  }
  if ( this->m_Elastix->GetFixedImage()->GetImageDimension() == 3 )
  {
    if ( this->m_Elastix->GetFixedImage()->GetLargestPossibleRegion().GetSize()[2] != 1 )
    {
      itkExceptionMacro( << "Metric can only be used for 2D-3D registration. FixedImageSize[2] must be 1" );
    }
  }

} // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void
GradientDifferenceMetric<TElastix>
::BeforeEachResolution( void )
{
  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  bool usescales = true;

  for ( unsigned int i = 0; i < MovingImageDimension; ++i )
  {
    usescales &= this->GetConfiguration()->ReadParameter(
    movingImageDerivativeScales[ i ], "MovingImageDerivativeScales",
    this->GetComponentLabel(), i, -1, false );
  }
  if ( usescales )
  {
    this->SetUseMovingImageDerivativeScales( true );
    this->SetMovingImageDerivativeScales( movingImageDerivativeScales );
    elxout << "Multiplying moving image derivatives by: "
      << movingImageDerivativeScales << std::endl;
  }

  typedef typename elastix::OptimizerBase<TElastix>::ITKBaseType::ScalesType  ScalesType;
  ScalesType scales = this->m_Elastix->GetElxOptimizerBase()->GetAsITKBaseType()->GetScales();
  this->SetScales( scales );

} // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxGradientDifferenceMetric_HXX__
