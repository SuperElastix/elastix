/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxRSGDEachParameterApart_hxx
#define __elxRSGDEachParameterApart_hxx

#include "elxRSGDEachParameterApart.h"
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
RSGDEachParameterApart< TElastix >::BeforeRegistration( void )
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
RSGDEachParameterApart< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Set the Gradient Magnitude Stopping Criterion.*/
  double minGradientMagnitude = 1e-8;
  this->m_Configuration->ReadParameter( minGradientMagnitude,
    "MinimumGradientMagnitude", this->GetComponentLabel(), level, 0 );
  this->SetGradientMagnitudeTolerance( minGradientMagnitude );

  /** Set the MaximumStepLength.*/
  double maxStepLength = 16.0 / pow( 2.0, static_cast< int >( level ) );
  this->m_Configuration->ReadParameter( maxStepLength,
    "MaximumStepLength", this->GetComponentLabel(), level, 0 );
  this->SetMaximumStepLength( maxStepLength );

  /** Set the MinimumStepLength.*/
  double minStepLength = 0.5 / pow( 2.0, static_cast< int >( level ) );
  this->m_Configuration->ReadParameter( minStepLength,
    "MinimumStepLength", this->GetComponentLabel(), level, 0 );
  this->SetMinimumStepLength( minStepLength );

  /** Set the Relaxation factor
   * \todo Implement this also in the itkRSGDEachParameterApartOptimizer
   *  (just like in the RegularStepGradientDescentOptimizer) and
   * uncomment the following:
   */
  //double relaxationFactor = 0.5;
  //this->m_Configuration->ReadParameter( relaxationFactor,
  //  "RelaxationFactor", this->GetComponentLabel(), level, 0 );
  //this->SetRelaxationFactor( relaxationFactor );

  /** \todo max and min steplength should maybe depend on the imagespacing or on something else... */

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 100;
  this->m_Configuration->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfIterations( maximumNumberOfIterations );

}   // end BeforeEachResolution


/**
 * ***************** AfterEachIteration *************************
 */

template< class TElastix >
void
RSGDEachParameterApart< TElastix >
::AfterEachIteration( void )
{
  /** Print some information */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetValue();
  xl::xout[ "iteration" ][ "3:StepSize" ] << this->GetCurrentStepLength();
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << this->GetGradientMagnitude();

}   // end AfterEachIteration


/**
 * ***************** AfterEachResolution *************************
 */

template< class TElastix >
void
RSGDEachParameterApart< TElastix >
::AfterEachResolution( void )
{

  /**
   * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
   * ImageNotAvailable, SamplesNotAvailable, MaximumNumberOfIterations, MetricError
   */
  std::string stopcondition;

  switch( this->GetStopCondition() )
  {

    case GradientMagnitudeTolerance:
      stopcondition = "Minimum gradient magnitude has been reached";
      break;

    case StepTooSmall:
      stopcondition = "Minimum step size has been reached";
      break;

    case MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case ImageNotAvailable:
      stopcondition = "No image available";
      break;

    case SamplesNotAvailable:
      stopcondition = "No samples available";
      break;

    case MetricError:
      stopcondition = "Error in metric";

    default:
      stopcondition = "Unknown";
      break;

  }
  /** Print the stopping condition */

  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

}   // end AfterEachResolution


/**
 * ******************* AfterRegistration ************************
 */

template< class TElastix >
void
RSGDEachParameterApart< TElastix >
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
RSGDEachParameterApart< TElastix >
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

#endif // end #ifndef __elxRSGDEachParameterApart_hxx
