/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxMutualInformationHistogramMetric_HXX__
#define __elxMutualInformationHistogramMetric_HXX__

#include "elxMutualInformationHistogramMetric.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
MutualInformationHistogramMetric< TElastix >
::MutualInformationHistogramMetric()
{}  // end Constructor

/**
 * ******************* Initialize ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >::Initialize( void ) throw ( itk::ExceptionObject )
{
  TimerPointer timer = TimerType::New();
  timer->StartTimer();
  this->Superclass1::Initialize();
  timer->StopTimer();
  elxout << "Initialization of MutualInformationHistogramMetric metric took: "
         << static_cast< long >( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

}   // end Initialize


/**
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >::BeforeRegistration( void )
{
  /** This exception can be removed once this class is fully implemented. */
  itkExceptionMacro( << "ERROR: This class is not yet fully implemented." );

}   // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
MutualInformationHistogramMetric< TElastix >::BeforeEachResolution( void )
{
  /** \todo adapt SecondOrderRegularisationMetric.
   * Set alpha, which balances the similarity and deformation energy
   * E_total = (1-alpha)*E_sim + alpha*E_def.
   * metric->SetAlpha( config.GetAlpha(level) );
   */

  const unsigned int nrOfParameters = this->m_Elastix->GetElxTransformBase()
    ->GetAsITKBaseType()->GetNumberOfParameters();
  ScalesType derivativeStepLengthScales( nrOfParameters );
  derivativeStepLengthScales.Fill( 1.0 );

  /** Read the parameters from the ParameterFile.*
  this->m_Configuration->ReadParameter( histogramSize, "HistogramSize", 0 );
  this->m_Configuration->ReadParameter( paddingValue, "PaddingValue", 0 );
  this->m_Configuration->ReadParameter( derivativeStepLength, "DerivativeStepLength", 0 );
  this->m_Configuration->ReadParameter( derivativeStepLengthScales, "DerivativeStepLengthScales", 0 );
  this->m_Configuration->ReadParameter( upperBoundIncreaseFactor, "UpperBoundIncreaseFactor", 0 );
  this->m_Configuration->ReadParameter( usePaddingValue, "UsePaddingValue", 0 );
  */
  /** Set them.*/
  //this->SetHistogramSize( ?? );
  //this->SetPaddingValue( ?? );
  //this->SetDerivativeStepLength( ?? );
  this->SetDerivativeStepLengthScales( derivativeStepLengthScales );
  //this->SetUpperBoundIncreaseFactor( ?? );
  //this->SetUsePaddingValue( ?? );

}   // end BeforeEachResolution


} // end namespace elastix

#endif // end #ifndef __elxMutualInformationHistogramMetric_HXX__
