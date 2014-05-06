/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxSimplex_hxx
#define __elxSimplex_hxx

#include "elxSimplex.h"
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
Simplex< TElastix >
::BeforeRegistration( void )
{
  /** Add the target cell "stepsize" to xout["iteration"].*/
  xout[ "iteration" ].AddTargetCell( "2:Metric" );
  xout[ "iteration" ].AddTargetCell( "3:StepSize" );
  xout[ "iteration" ].AddTargetCell( "4:||Gradient||" );

  /** Format the metric and stepsize as floats */
  xl::xout[ "iteration" ][ "2:Metric" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "3:StepSize" ] << std::showpoint << std::fixed;
  xl::xout[ "iteration" ][ "4:||Gradient||" ] << std::showpoint << std::fixed;

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template< class TElastix >
void
Simplex< TElastix >
::BeforeEachResolution( void )
{
  /** Get the current resolution level.*/
  unsigned int level = static_cast< unsigned int >(
    this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );

  /** Set the value tolerance.*/
  double valueTolerance = 1e-8;
  this->m_Configuration->ReadParameter( valueTolerance,
    "ValueTolerance", this->GetComponentLabel(), level, 0 );
  this->SetFunctionConvergenceTolerance( valueTolerance );

  /** Set the maximumNumberOfIterations.*/
  unsigned int maximumNumberOfIterations = 500;
  this->m_Configuration->ReadParameter( maximumNumberOfIterations,
    "MaximumNumberOfIterations", this->GetComponentLabel(), level, 0 );
  this->SetMaximumNumberOfIterations( maximumNumberOfIterations );

  /** Set the automaticinitialsimplex.*/
  bool automaticinitialsimplex = false;
  this->m_Configuration->ReadParameter( automaticinitialsimplex,
    "AutomaticInitialSimplex", this->GetComponentLabel(), level, 0 );
  this->SetAutomaticInitialSimplex( automaticinitialsimplex );

  /** If no automaticinitialsimplex, InitialSimplexDelta should be given.*/
  if( !automaticinitialsimplex )
  {
    unsigned int numberofparameters
      = this->m_Elastix->GetElxTransformBase()->GetAsITKBaseType()->GetNumberOfParameters();
    ParametersType initialsimplexdelta( numberofparameters );
    initialsimplexdelta.Fill( 1 );

    for( unsigned int i = 0; i < numberofparameters; i++ )
    {
      this->m_Configuration->ReadParameter(
        initialsimplexdelta[ i ], "InitialSimplexDelta", i );
    }

    this->SetInitialSimplexDelta( initialsimplexdelta );
  }

} // end BeforeEachResolution()


/**
 * ***************** AfterEachIteration *************************
 */

template< class TElastix >
void
Simplex< TElastix >
::AfterEachIteration( void )
{
  /** Print some information */
  xl::xout[ "iteration" ][ "2:Metric" ] << this->GetCachedValue();
  //xl::xout["iteration"]["3:StepSize"] << this->GetStepLength();

} // end AfterEachIteration()


/**
 * ***************** AfterEachResolution *************************
 */

template< class TElastix >
void
Simplex< TElastix >
::AfterEachResolution( void )
{
  /**
  * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
  * ImageNotAvailable, CostFunctionError, MaximumNumberOfIterations
  */
  std::string stopcondition = this->GetStopConditionDescription();

  /** Print the stopping condition */
  elxout << "Stopping condition: " << stopcondition << "." << std::endl;

} // end AfterEachResolution()


/**
 * ******************* AfterRegistration ************************
 */

template< class TElastix >
void
Simplex< TElastix >
::AfterRegistration( void )
{
  /** Print the best metric value */
  //double bestValue = this->GetValue();
  double bestValue = this->GetCachedValue();
  elxout << std::endl << "Final metric value  = " << bestValue  << std::endl;

} // end AfterRegistration()


/**
 * ******************* SetInitialPosition ***********************
 */

template< class TElastix >
void
Simplex< TElastix >
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

} // end SetInitialPosition()


} // end namespace elastix

#endif // end #ifndef __elxSimplex_hxx
