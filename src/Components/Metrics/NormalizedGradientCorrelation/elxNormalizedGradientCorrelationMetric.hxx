/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __elxNormalizedGradientCorrelationMetric_HXX__
#define __elxNormalizedGradientCorrelationMetric_HXX__

#include "elxNormalizedGradientCorrelationMetric.h"


namespace elastix
{
using namespace itk;

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
NormalizedGradientCorrelationMetric<TElastix>
::Initialize( void ) throw ( ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of NormalizedGradientCorrelation metric took: "
    << static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void NormalizedGradientCorrelationMetric<TElastix>
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

} // end BeforeEachResolution()


} // end namespace elastix

#endif // end #ifndef __elxNormalizedGradientCorrelationMetric_HXX__
