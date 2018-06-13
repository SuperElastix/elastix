/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxStandardGradientDescent_hxx
#define __elxStandardGradientDescent_hxx

#include "elxStandardGradientDescent.h"
#include <iomanip>
#include <string>

namespace elastix
{

/**
 * ***************** Constructor ***********************
 */

template< class TElastix >
StandardGradientDescent< TElastix >::StandardGradientDescent()
{
  this->m_MaximumNumberOfSamplingAttempts = 0;
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration        = 0;

}   // end Constructor()


/**
 * ***************** BeforeRegistration ***********************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"].*/
  xout[ "iteration" ].AddTargetCell( "2:Metric" );
  xout[ "iteration" ].AddTargetCell( "3:StepSize" );
  xout[ "iteration" ].AddTargetCell( "4:||Gradient||" );

  /** Format the metric and stepsize as floats */
  xl::xout[ "iteration" ][ "2:Metric" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3:StepSize" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << std::showpoint << std::fixed;

}   // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Set the maximumNumberOfIterations. */
  unsigned int maximumNumberOfIterations = 500;
  this->GetConfiguration()->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetNumberOfIterations( maximumNumberOfIterations );

  /** Set the gain parameters */
  double a     = 400.0;
  double A     = 50.0;
  double alpha = 0.602;

  this->GetConfiguration()->ReadParameter( a, "SP_a", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( A, "SP_A", this->GetComponentLabel(), level, 0 );
  this->GetConfiguration()->ReadParameter( alpha, "SP_alpha", this->GetComponentLabel(), level, 0 );

  this->SetParam_a( a );
  this->SetParam_A( A );
  this->SetParam_alpha( alpha );

  /** Set the MaximumNumberOfSamplingAttempts. */
  unsigned int maximumNumberOfSamplingAttempts = 0;
  this->GetConfiguration()->ReadParameter( maximumNumberOfSamplingAttempts,
    "MaximumNumberOfSamplingAttempts", this->GetComponentLabel(), level, 0 );
  this->SetMaximumNumberOfSamplingAttempts( maximumNumberOfSamplingAttempts );
  if( maximumNumberOfSamplingAttempts > 5 )
  {
    elxout[ "warning" ]
      << "\nWARNING: You have set MaximumNumberOfSamplingAttempts to "
      << maximumNumberOfSamplingAttempts << ".\n"
      << "  This functionality is known to cause problems (stack overflow) for large values.\n"
      << "  If elastix stops or segfaults for no obvious reason, reduce this value.\n"
      << "  You may select the RandomSparseMask image sampler to fix mask-related problems.\n"
      << std::endl;
  }

}   // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::AfterEachIteration( void )
{
  /** Print some information */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetValue();
  xl::xout[ "iteration" ][ "3:StepSize" ] << this->GetLearningRate();
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << this->GetGradient().magnitude();

  /** Select new spatial samples for the computation of the metric */
  if( this->GetNewSamplesEveryIteration() )
  {
    this->SelectNewSamples();
  }

}   // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::AfterEachResolution( void )
{
  /**
   * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }
   */
  std::string stopcondition;
  switch( this->GetStopCondition() )
  {

    case MaximumNumberOfIterations:
      stopcondition = "Maximum number of iterations has been reached";
      break;

    case MetricError:
      stopcondition = "Error in metric";
      break;

    default:
      stopcondition = "Unknown";
      break;

  }

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

}   // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::AfterRegistration( void )
{
  /** Print the best metric value */
  double bestValue = this->GetValue();
  elxout
    << std::endl
    << "Final metric value  = "
    << bestValue
    << std::endl;

}   // end AfterRegistration()


/**
 * ****************** StartOptimization *************************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::StartOptimization( void )
{
  /** Check if the entered scales are correct and != [ 1 1 1 ...] */
  this->SetUseScales( false );
  const ScalesType & scales = this->GetScales();
  if( scales.GetSize() == this->GetInitialPosition().GetSize() )
  {
    ScalesType unit_scales( scales.GetSize() );
    unit_scales.Fill( 1.0 );
    if( scales != unit_scales )
    {
      /** only then: */
      this->SetUseScales( true );
    }
  }

  /** Reset these values. */
  this->m_CurrentNumberOfSamplingAttempts = 0;
  this->m_PreviousErrorAtIteration        = 0;

  /** Superclass implementation. */
  this->Superclass1::StartOptimization();

}   // end StartOptimization()


/**
 * ****************** MetricErrorResponse *************************
 */

template< class TElastix >
void
StandardGradientDescent< TElastix >
::MetricErrorResponse( itk::ExceptionObject & err )
{
  if( this->GetCurrentIteration() != this->m_PreviousErrorAtIteration )
  {
    this->m_PreviousErrorAtIteration        = this->GetCurrentIteration();
    this->m_CurrentNumberOfSamplingAttempts = 1;
  }
  else
  {
    this->m_CurrentNumberOfSamplingAttempts++;
  }

  if( this->m_CurrentNumberOfSamplingAttempts <= this->m_MaximumNumberOfSamplingAttempts )
  {
    this->SelectNewSamples();
    this->ResumeOptimization();
  }
  else
  {
    /** Stop optimisation and pass on exception. */
    this->Superclass1::MetricErrorResponse( err );
  }

}   // end MetricErrorResponse()


} // end namespace elastix

#endif // end #ifndef __elxStandardGradientDescent_hxx
