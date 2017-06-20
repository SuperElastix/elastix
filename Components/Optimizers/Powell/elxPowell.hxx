/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __elxPowell_hxx
#define __elxPowell_hxx

#include "elxPowell.h"
#include <iomanip>
#include <string>
#include "vnl/vnl_math.h"

namespace elastix
{

/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
Powell< TElastix >::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"].*/
  xout[ "iteration" ].AddTargetCell( "2:Metric" );
  xout[ "iteration" ].AddTargetCell( "3:StepSize" );
  xout[ "iteration" ].AddTargetCell( "4:||Gradient||" );

  /** Format the metric and stepsize as floats */
  xl::xout[ "iteration" ][ "2:Metric" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3:StepSize" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << std::showpoint << std::fixed;

}   // end BeforeRegistration


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
Powell< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Set the value tolerance.*/
  double valueTolerance = 1e-8;
  this->m_Configuration->ReadParameter( valueTolerance,
    "ValueTolerance", this->GetComponentLabel(), level, 0 );
  this->SetValueTolerance( valueTolerance );

  /** Set the MaximumStepLength.*/
  double maxStepLength = 16.0 / vcl_pow( 2.0, static_cast< int >( level ) );
  this->m_Configuration->ReadParameter( maxStepLength,
    "MaximumStepLength", this->GetComponentLabel(), level, 0 );
  this->SetStepLength( maxStepLength );

  /** Set the MinimumStepLength.*/
  double stepTolerance = 0.5 / vcl_pow( 2.0, static_cast< int >( level ) );
  this->m_Configuration->ReadParameter( stepTolerance,
    "StepTolerance", this->GetComponentLabel(), level, 0 );
  this->SetStepTolerance( stepTolerance );

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetMaximumIteration( maximumNumberOfIterations );

}   // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template< class TElastix >
void
Powell< TElastix >
::AfterEachIteration( void )
{
  /** Print some information */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetValue();
  xl::xout[ "iteration" ][ "3:StepSize" ] << this->GetStepLength();
}   // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template< class TElastix >
void
Powell< TElastix >
::AfterEachResolution( void )
{
  /**
   * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
   * ImageNotAvailable, CostFunctionError, MaximumNumberOfIterations
   */
  std::string stopcondition = this->GetStopConditionDescription();

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

}   // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template< class TElastix >
void
Powell< TElastix >
::AfterRegistration( void )
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  elxout << std::endl << "Final metric value  = " << bestValue  << std::endl;

}   // end AfterRegistration


/**
 * ******************* SetInitialPosition ***********************
 */

template< class TElastix >
void
Powell< TElastix >
::SetInitialPosition( const ParametersType & param )
{
  /** Override the implementation in itkOptimizer.h, to
   * ensure that the scales array and the parameters
   * array have the same size.
   */

  /** Call the Superclass' implementation. */
  this->Superclass1::SetInitialPosition( param );

  /** Set the scales array to the same size if the size has been changed */
  ScalesType   scales    = this->GetScales();
  unsigned int paramsize = param.Size();

  if( ( scales.Size() ) != paramsize )
  {
    ScalesType newscales( paramsize );
    newscales.Fill( 1.0 );
    this->SetScales( newscales );
  }

  /** \todo to optimizerbase? */

}   // end SetInitialPosition


} // end namespace elastix

#endif // end #ifndef __elxPowell_hxx
