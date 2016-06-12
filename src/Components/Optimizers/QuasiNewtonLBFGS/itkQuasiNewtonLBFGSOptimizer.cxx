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

#ifndef __itkQuasiNewtonLBFGSOptimizer_cxx
#define __itkQuasiNewtonLBFGSOptimizer_cxx

#include "itkQuasiNewtonLBFGSOptimizer.h"
#include "itkArray.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ******************** Constructor *************************
 */

QuasiNewtonLBFGSOptimizer::QuasiNewtonLBFGSOptimizer()
{
  itkDebugMacro( "Constructor" );

  this->m_CurrentValue      = NumericTraits< MeasureType >::Zero;
  this->m_CurrentIteration  = 0;
  this->m_StopCondition     = Unknown;
  this->m_Stop              = false;
  this->m_CurrentStepLength = 0.0;
  this->m_InLineSearch      = false;
  this->m_Point             = 0;
  this->m_PreviousPoint     = 0;
  this->m_Bound             = 0;

  this->m_MaximumNumberOfIterations  = 100;
  this->m_GradientMagnitudeTolerance = 1e-5;
  this->m_LineSearchOptimizer        = 0;
  this->m_Memory                     = 5;

}   // end constructor


/**
 * ******************* StartOptimization *********************
 */

void
QuasiNewtonLBFGSOptimizer::StartOptimization()
{
  itkDebugMacro( "StartOptimization" );

  /** Reset some variables */
  this->m_Point             = 0;
  this->m_PreviousPoint     = 0;
  this->m_Bound             = 0;
  this->m_Stop              = false;
  this->m_StopCondition     = Unknown;
  this->m_CurrentIteration  = 0;
  this->m_CurrentStepLength = 0.0;
  this->m_CurrentValue      = NumericTraits< MeasureType >::Zero;

  /** Get the number of parameters; checks also if a cost function has been set at all.
  * if not: an exception is thrown */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Set the current gradient to (0 0 0 ...) */
  this->m_CurrentGradient.SetSize( numberOfParameters );
  this->m_CurrentGradient.Fill( 0.0 );

  /** Resize Rho, Alpha, S and Y. */
  this->m_Rho.SetSize( this->GetMemory() );
  this->m_S.resize( this->GetMemory() );
  this->m_Y.resize( this->GetMemory() );

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition( this->GetInitialPosition() );

  if( !this->m_Stop )
  {
    this->ResumeOptimization();
  }

}   // end StartOptimization


/**
 * ******************* ResumeOptimization *********************
 */

void
QuasiNewtonLBFGSOptimizer::ResumeOptimization()
{
  itkDebugMacro( "ResumeOptimization" );

  this->m_Stop              = false;
  this->m_StopCondition     = Unknown;
  this->m_CurrentStepLength = 0.0;

  ParametersType searchDir;
  DerivativeType previousGradient;

  this->InvokeEvent( StartEvent() );

  /** Get initial value and derivative */
  try
  {
    this->GetScaledValueAndDerivative(
      this->GetScaledCurrentPosition(),
      this->m_CurrentValue,
      this->m_CurrentGradient );
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

  /** Test if the gradient was not zero already by chance */
  bool convergence = this->TestConvergence( false );
  if( convergence )
  {
    this->StopOptimization();
  }

  /** Start iterating */
  while( !this->m_Stop )
  {
    /** Compute the new search direction, using the current gradient */
    this->ComputeSearchDirection( this->GetCurrentGradient(), searchDir );

    if( this->m_Stop )
    {
      break;
    }

    /** Store the current gradient */
    previousGradient = this->GetCurrentGradient();

    /** Perform a line search along the search direction. On return the
     * m_CurrentStepLength, m_CurrentScaledPosition, m_CurrentValue, and
     * m_CurrentGradient are updated. */
    this->LineSearch(
      searchDir,
      this->m_CurrentStepLength,
      this->m_ScaledCurrentPosition,
      this->m_CurrentValue,
      this->m_CurrentGradient );

    if( this->m_Stop )
    {
      break;
    }

    /** Store s (in m_S), y (in m_Y), and ys (in m_Rho). These are used to
     * compute the search direction in the next iterations */
    if( this->GetMemory() > 0 )
    {
      ParametersType s;
      DerivativeType y;
      s = this->GetCurrentStepLength() * searchDir;
      y = this->GetCurrentGradient() - previousGradient;
      this->StoreCurrentPoint( s, y );
      s.clear();
      y.clear();
    }

    /** Number of valid entries in m_S and m_Y */
    if( this->m_Bound < this->GetMemory() )
    {
      this->m_Bound++;
    }

    this->InvokeEvent( IterationEvent() );

    if( this->m_Stop )
    {
      break;
    }

    /** Test if convergence has occurred */
    convergence = this->TestConvergence( true );
    if( convergence )
    {
      this->StopOptimization();
      break;
    }

    /** Update the index of m_S and m_Y for the next iteration */
    this->m_PreviousPoint = this->m_Point;
    this->m_Point++;
    if( this->m_Point >= this->m_Memory )
    {
      this->m_Point = 0;
    }

    this->m_CurrentIteration++;

  }   // end while !m_Stop

}   // end ResumeOptimization


/**
 * *********************** StopOptimization *****************************
 */

void
QuasiNewtonLBFGSOptimizer::StopOptimization()
{
  itkDebugMacro( "StopOptimization" );
  this->m_Stop = true;
  this->InvokeEvent( EndEvent() );
}   // end StopOptimization()


/**
 * ********************* ComputeDiagonalMatrix ********************
 */

void
QuasiNewtonLBFGSOptimizer::ComputeDiagonalMatrix( DiagonalMatrixType & diag_H0 )
{
  diag_H0.SetSize(
    this->GetScaledCostFunction()->GetNumberOfParameters() );

  double fill_value = 1.0;

  if( this->m_Bound > 0 )
  {
    const DerivativeType & y  = this->m_Y[ this->m_PreviousPoint ];
    const double           ys = 1.0 / this->m_Rho[ this->m_PreviousPoint ];
    const double           yy = y.squared_magnitude();
    fill_value = ys / yy;
    if( fill_value <= 0. )
    {
      this->m_StopCondition = InvalidDiagonalMatrix;
      this->StopOptimization();
    }
  }

  diag_H0.Fill( fill_value );

}   // end ComputeDiagonalMatrix


/**
 * *********************** ComputeSearchDirection ************************
 */

void
QuasiNewtonLBFGSOptimizer::ComputeSearchDirection(
  const DerivativeType & gradient,
  ParametersType & searchDir )
{
  itkDebugMacro( "ComputeSearchDirection" );

  /** Assumes m_Rho, m_S, and m_Y are up-to-date at m_PreviousPoint */

  typedef Array< double > AlphaType;
  AlphaType alpha( this->GetMemory() );

  const unsigned int numberOfParameters = gradient.GetSize();
  DiagonalMatrixType H0;
  this->ComputeDiagonalMatrix( H0 );

  searchDir = -gradient;

  int cp = static_cast< int >( this->m_Point );

  for( unsigned int i = 0; i < this->m_Bound; ++i )
  {
    --cp;
    if( cp == -1 )
    {
      cp = this->GetMemory() - 1;
    }
    const double sq = inner_product( this->m_S[ cp ], searchDir );
    alpha[ cp ] = this->m_Rho[ cp ] * sq;
    const double &         alpha_cp = alpha[ cp ];
    const DerivativeType & y        = this->m_Y[ cp ];
    for( unsigned int j = 0; j < numberOfParameters; ++j )
    {
      searchDir[ j ] -= alpha_cp * y[ j ];
    }
  }

  for( unsigned int j = 0; j < numberOfParameters; ++j )
  {
    searchDir[ j ] *= H0[ j ];
  }

  for( unsigned int i = 0; i < this->m_Bound; ++i )
  {
    const double           yr             = inner_product( this->m_Y[ cp ], searchDir );
    const double           beta           = this->m_Rho[ cp ] * yr;
    const double           alpha_min_beta = alpha[ cp ] - beta;
    const ParametersType & s              = this->m_S[ cp ];
    for( unsigned int j = 0; j < numberOfParameters; ++j )
    {
      searchDir[ j ] += alpha_min_beta * s[ j ];
    }
    ++cp;
    if( static_cast< unsigned int >( cp ) == this->GetMemory() )
    {
      cp = 0;
    }
  }

  /** Normalize if no information about previous steps is available yet */
  if( this->m_Bound == 0 )
  {
    searchDir /= gradient.magnitude();
  }

}   // end ComputeSearchDirection


/**
 * ********************* LineSearch *******************************
 *
 * Perform a line search along the search direction. On return the
 * step, x (new position), f (value at x), and g (derivative at x)
 * are updated.
 */

void
QuasiNewtonLBFGSOptimizer::LineSearch(
  const ParametersType searchDir,
  double & step,
  ParametersType & x,
  MeasureType & f,
  DerivativeType & g )
{

  itkDebugMacro( "LineSearch" );

  LineSearchOptimizerPointer LSO = this->GetLineSearchOptimizer();

  if( LSO.IsNull() )
  {
    this->m_StopCondition = LineSearchError;
    this->StopOptimization();
    itkExceptionMacro( << "No line search optimizer set" );
  }

  LSO->SetCostFunction( this->m_ScaledCostFunction );
  LSO->SetLineSearchDirection( searchDir );
  LSO->SetInitialPosition( x );
  LSO->SetInitialValue( f );
  LSO->SetInitialDerivative( g );

  this->SetInLineSearch( true );
  try
  {
    LSO->StartOptimization();
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = LineSearchError;
    this->StopOptimization();
    throw err;
  }
  this->SetInLineSearch( false );

  step = LSO->GetCurrentStepLength();
  x    = LSO->GetCurrentPosition();

  try
  {
    LSO->GetCurrentValueAndDerivative( f, g );
  }
  catch( ExceptionObject & err )
  {
    this->m_StopCondition = MetricError;
    this->StopOptimization();
    throw err;
  }

}   // end LineSearch


/**
 * ********************* StoreCurrentPoint ************************
 */

void
QuasiNewtonLBFGSOptimizer::StoreCurrentPoint(
  const ParametersType & step,
  const DerivativeType & grad_dif )
{
  itkDebugMacro( "StoreCurrentPoint" );

  this->m_S[ this->m_Point ]   = step;                                  // s
  this->m_Y[ this->m_Point ]   = grad_dif;                              // y
  this->m_Rho[ this->m_Point ] = 1.0 / inner_product( step, grad_dif ); // 1/ys

}   // end StoreCurrentPoint


/**
 * ********************* TestConvergence ************************
 */

bool
QuasiNewtonLBFGSOptimizer::TestConvergence( bool firstLineSearchDone )
{
  itkDebugMacro( "TestConvergence" );

  /** Check for zero step length */
  if( firstLineSearchDone )
  {
    if( this->m_CurrentStepLength < NumericTraits< double >::epsilon() )
    {
      this->m_StopCondition = ZeroStep;
      return true;
    }
  }

  /** Check if the maximum number of iterations will not be exceeded in the following iteration */
  if( ( this->GetCurrentIteration() + 1 ) >= this->GetMaximumNumberOfIterations() )
  {
    this->m_StopCondition = MaximumNumberOfIterations;
    return true;
  }

  /** Check for convergence of gradient magnitude */
  const double gnorm = this->GetCurrentGradient().magnitude();
  const double xnorm = this->GetScaledCurrentPosition().magnitude();
  if( gnorm / vnl_math_max( 1.0, xnorm ) <= this->GetGradientMagnitudeTolerance() )
  {
    this->m_StopCondition = GradientMagnitudeTolerance;
    return true;
  }

  return false;

}   // end TestConvergence


} // end namespace itk

#endif // #ifndef __itkQuasiNewtonLBFGSOptimizer_cxx
