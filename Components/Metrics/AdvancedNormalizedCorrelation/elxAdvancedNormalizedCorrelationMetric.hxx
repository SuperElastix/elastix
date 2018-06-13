/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxAdvancedNormalizedCorrelationMetric_HXX__
#define __elxAdvancedNormalizedCorrelationMetric_HXX__

#include "elxAdvancedNormalizedCorrelationMetric.h"

namespace elastix
{

/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
AdvancedNormalizedCorrelationMetric< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = ( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
  this->SetCurrentResolutionLevel( level );

  /** Get and set SubtractMean. Default true. */
  bool subtractMean = true;
  this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean",
    this->GetComponentLabel(), level, 0 );
  this->SetSubtractMean( subtractMean );

  /** Set moving image derivative scales. */
  this->SetUseMovingImageDerivativeScales( false );
  MovingImageDerivativeScalesType movingImageDerivativeScales;
  movingImageDerivativeScales.Fill( 1.0 );
  bool usescales = true;
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

  /** Set grid shift strategy. */
  string GridShiftStrategy = "";
  this->GetConfiguration()->ReadParameter( GridShiftStrategy,
	  "GridShiftStrategy", this->GetComponentLabel(), 0, 0 );
  this->SetGridShiftStrategy( GridShiftStrategy );

  /** Set grid shift step number. */
  unsigned int GridShiftStepNumber = 0;
  this->GetConfiguration()->ReadParameter( GridShiftStepNumber,
	  "GridShiftStepNumber", this->GetComponentLabel(), 0, 0 );
  this->SetGridShiftStepNumber( GridShiftStepNumber );

  /** Set use full perturbation range. */
  bool UseFullPerturbationRange = false;
  this->GetConfiguration()->ReadParameter( UseFullPerturbationRange,
	  "UseFullPerturbationRange", this->GetComponentLabel(), 0, 0 );
  this->SetUseFullPerturbationRange( UseFullPerturbationRange );

  /** Set perturbation factor. */
  unsigned int perturbationFactor = 1;
  this->GetConfiguration()->ReadParameter( perturbationFactor,
	  "PerturbationFactor", this->GetComponentLabel(), 0, 0 );
  this->SetPerturbationFactor( perturbationFactor );
}   // end BeforeEachResolution()


/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
AdvancedNormalizedCorrelationMetric< TElastix >
::Initialize( void ) throw ( itk::ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of AdvancedNormalizedCorrelation metric took: "
         << static_cast< long >( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

}   // end Initialize()

/** Switch grid shift strategy. */
template <class TElastix>
void AdvancedNormalizedCorrelationMetric<TElastix>
::SetUseGridShiftStrategy( bool useStrategy )
{
	this->SetUseGridShift( useStrategy );
}

/** Initialize random shift list. */
template <class TElastix>
void AdvancedNormalizedCorrelationMetric<TElastix>
::SetRandomShiftList( std::vector< double > randomList )
{
	this->InitializeRandomShiftList(randomList);
}

} // end namespace elastix

#endif // end #ifndef __elxAdvancedNormalizedCorrelationMetric_HXX__
