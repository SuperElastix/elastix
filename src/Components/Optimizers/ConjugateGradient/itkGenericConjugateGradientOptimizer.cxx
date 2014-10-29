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

#ifndef __itkGenericConjugateGradientOptimizer_cxx
#define __itkGenericConjugateGradientOptimizer_cxx

#include "itkGenericConjugateGradientOptimizer.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ******************** Constructor *************************
 */

GenericConjugateGradientOptimizer::GenericConjugateGradientOptimizer()
{
  itkDebugMacro( "Constructor" );

  this->m_CurrentValue                          = NumericTraits< MeasureType >::Zero;
  this->m_CurrentIteration                      = 0;
  this->m_StopCondition                         = Unknown;
  this->m_Stop                                  = false;
  this->m_CurrentStepLength                     = 0.0;
  this->m_InLineSearch                          = false;
  this->m_MaximumNumberOfIterations             = 100;
  this->m_ValueTolerance                        = 1e-5;
  this->m_GradientMagnitudeTolerance            = 1e-5;
  this->m_MaxNrOfItWithoutImprovement           = 10;
  this->m_UseDefaultMaxNrOfItWithoutImprovement = true;
  this->m_LineSearchOptimizer                   = 0;
  this->m_PreviousGradientAndSearchDirValid     = false;

  this->AddBetaDefinition(
    "SteepestDescent", &Self::ComputeBetaSD );
  this->AddBetaDefinition(
    "FletcherReeves", &Self::ComputeBetaFR );
  this->AddBetaDefinition(
    "PolakRibiere", &Self::ComputeBetaPR );
  this->AddBetaDefinition(
    "DaiYuan", &Self::ComputeBetaDY );
  this->AddBetaDefinition(
    "HestenesStiefel", &Self::ComputeBetaHS );
  this->AddBetaDefinition(
    "DaiYuanHestenesStiefel", &Self::ComputeBetaDYHS );

  this->SetBetaDefinition( "DaiYuanHestenesStiefel" );

}   // end constructor


/**
 * ******************* StartOptimization *********************
 */

void
GenericConjugateGradientOptimizer::StartOptimization()
{
  itkDebugMacro( "StartOptimization" );

  /** Reset some variables */
  this->m_Stop                              = false;
  this->m_StopCondition                     = Unknown;
  this->m_CurrentIteration                  = 0;
  this->m_CurrentStepLength                 = 0.0;
  this->m_CurrentValue                      = NumericTraits< MeasureType >::Zero;
  this->m_PreviousGradientAndSearchDirValid = false;

  /** Get the number of parameters; checks also if a cost function has been set at all.
  * if not: an exception is thrown */
  const unsigned int numberOfParameters
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  if( this->m_UseDefaultMaxNrOfItWithoutImprovement )
  {
    this->m_MaxNrOfItWithoutImprovement = numberOfParameters;
  }

  /** Set the current gradient to (0 0 0 ...) */
  this->m_CurrentGradient.SetSize( numberOfParameters );
  this->m_CurrentGradient.Fill( 0.0 );

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
GenericConjugateGradientOptimizer::ResumeOptimization()
{
  itkDebugMacro( "ResumeOptimization" );

  this->m_Stop                              = false;
  this->m_StopCondition                     = Unknown;
  this->m_PreviousGradientAndSearchDirValid = false;
  const double TINY_NUMBER = 1e-20;
  unsigned int limitCount  = 0;

  ParametersType searchDir;
  ParametersType previousSearchDir;
  DerivativeType previousGradient;
  MeasureType    previousValue;

  this->InvokeEvent( StartEvent() );

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

  /** Test if not by chance we are already converged */
  bool convergence = false;
  convergence = this->TestConvergence( false );
  if( convergence )
  {
    this->StopOptimization();
  }

  /** Start iterating */
  while( !this->m_Stop )
  {
    /** Store the current search direction */
    previousSearchDir = searchDir;

    /** Compute the new search direction */
    this->ComputeSearchDirection(
      previousGradient,
      this->GetCurrentGradient(),
      searchDir );

    if( this->m_Stop )
    {
      break;
    }

    /** Store the current gradient */
    previousGradient                          = this->GetCurrentGradient();
    previousValue                             = this->GetCurrentValue();
    this->m_PreviousGradientAndSearchDirValid = true;

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

    this->InvokeEvent( IterationEvent() );

    if( this->m_Stop )
    {
      break;
    }

    /** Check for convergence
     * \todo: move this code to TestConvergence() */
    if( 2.0 * vcl_abs( this->GetCurrentValue() - previousValue ) <=
      this->GetValueTolerance()
      * ( vcl_abs( this->GetCurrentValue() )
      + vcl_abs( previousValue ) + TINY_NUMBER ) )
    {
      if( limitCount < this->GetMaxNrOfItWithoutImprovement() )
      {
        //this->m_CurrentGradient[limitCount] = 1.0;
        // \todo gives errors (way to large gradient), should update
        // initial steplength estimate maybe
        limitCount++;
      }
      else
      {
        this->m_StopCondition = ValueTolerance;
        this->StopOptimization();
        break;
      }
    }
    else
    {
      limitCount = 0;
    }

    /** Test if convergence has occured in some other sense */
    convergence = false;
    convergence = this->TestConvergence( true );
    if( convergence )
    {
      this->StopOptimization();
      break;
    }

    /** Next iteration */
    this->m_CurrentIteration++;

  }   // end while !m_Stop

}   // end ResumeOptimization


/**
 * *********************** StopOptimization *****************************
 */

void
GenericConjugateGradientOptimizer::StopOptimization()
{
  itkDebugMacro( "StopOptimization" );
  this->m_Stop = true;
  this->InvokeEvent( EndEvent() );
}   // end StopOptimization()


/**
 * *********************** ComputeSearchDirection ************************
 */

void
GenericConjugateGradientOptimizer::ComputeSearchDirection(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  ParametersType & searchDir )
{
  itkDebugMacro( "ComputeSearchDirection" );

  const unsigned int numberOfParameters = gradient.GetSize();

  /** When no previous gradient and/or previous search direction are
   * available, return the negative gradient as search direction */
  if( !this->m_PreviousGradientAndSearchDirValid )
  {
    searchDir = -gradient;
    return;
  }

  /** Compute \beta, based on the previousGradient, the current gradient,
   * and the previous search direction */
  double beta = this->ComputeBeta( previousGradient, gradient, searchDir );

  if( this->m_Stop )
  {
    return;
  }

  /** Compute the new search direction */
  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    searchDir[ i ] = -gradient[ i ] + beta * searchDir[ i ];
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
GenericConjugateGradientOptimizer::LineSearch(
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

  /** For the next iteration: */
  //LSO->SetInitialStepLengthEstimate(step); for now in elx.

}   // end LineSearch


/**
 * *********************** ComputeBeta ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBeta(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & previousSearchDir )
{

  ComputeBetaFunctionType betaComputer
    = this->m_BetaDefinitionMap[ this->GetBetaDefinition() ];

  return ( ( *this ).*betaComputer )(
    previousGradient, gradient, previousSearchDir );

}   // end ComputeBeta


/**
 * ********************** ComputeBetaSD ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaSD(
  const DerivativeType & itkNotUsed( previousGradient ),
  const DerivativeType & itkNotUsed( gradient ),
  const ParametersType & itkNotUsed( previousSearchDir ) )
{
  /** A simple hack that makes the conjugate gradient equal to
   * a steepest descent method */
  return 0.0;
}   // end ComputeBetaSD


/**
 * ********************** ComputeBetaFR ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaFR(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & itkNotUsed( previousSearchDir ) )
{
  const unsigned int numberOfParameters = gradient.GetSize();
  double             num                = 0.0;
  double             den                = 0.0;

  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    const double & grad     = gradient[ i ];
    const double & prevgrad = previousGradient[ i ];
    num += grad * grad;
    den += prevgrad * prevgrad;
  }

  if( den <= NumericTraits< double >::epsilon() )
  {
    this->m_StopCondition = InfiniteBeta;
    this->StopOptimization();
    return 0.0;
  }
  return num / den;

}   // end ComputeBetaFR


/**
 * ********************** ComputeBetaPR ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaPR(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & itkNotUsed( previousSearchDir ) )
{
  const unsigned int numberOfParameters = gradient.GetSize();
  double             num                = 0.0;
  double             den                = 0.0;

  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    const double & grad     = gradient[ i ];
    const double & prevgrad = previousGradient[ i ];
    num += grad * ( grad - prevgrad );
    den += prevgrad * prevgrad;
  }

  if( den <= NumericTraits< double >::epsilon() )
  {
    this->m_StopCondition = InfiniteBeta;
    this->StopOptimization();
    return 0.0;
  }
  return num / den;

}   // end ComputeBetaPR


/**
 * ********************** ComputeBetaDY ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaDY(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & previousSearchDir )
{
  const unsigned int numberOfParameters = gradient.GetSize();
  double             num                = 0.0;
  double             den                = 0.0;

  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    const double & grad = gradient[ i ];
    num += grad * grad;
    den += previousSearchDir[ i ] * ( grad - previousGradient[ i ] );
  }

  if( den <= NumericTraits< double >::epsilon() )
  {
    this->m_StopCondition = InfiniteBeta;
    this->StopOptimization();
    return 0.0;
  }
  return num / den;
}   // end ComputeBetaDY


/**
 * ********************** ComputeBetaHS ******************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaHS(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & previousSearchDir )
{
  const unsigned int numberOfParameters = gradient.GetSize();
  double             num                = 0.0;
  double             den                = 0.0;

  for( unsigned int i = 0; i < numberOfParameters; ++i )
  {
    const double & diff = gradient[ i ] - previousGradient[ i ];
    num += gradient[ i ] * diff;
    den += previousSearchDir[ i ] * diff;
  }

  if( den <= NumericTraits< double >::epsilon() )
  {
    this->m_StopCondition = InfiniteBeta;
    this->StopOptimization();
    return 0.0;
  }

  return num / den;
}   // end ComputeBetaHS


/**
 * ********************** ComputeBetaDYHS ***************************
 */

double
GenericConjugateGradientOptimizer::ComputeBetaDYHS(
  const DerivativeType & previousGradient,
  const DerivativeType & gradient,
  const ParametersType & previousSearchDir )
{
  const double beta_DY = this->ComputeBetaDY(
    previousGradient, gradient, previousSearchDir );

  const double beta_HS = this->ComputeBetaHS(
    previousGradient, gradient, previousSearchDir );

  return vnl_math_max( 0.0, vnl_math_min( beta_DY, beta_HS ) );

}   // end ComputeBetaDYHS


/**
 * *********************** SetBetaDefinition **************************
 */

void
GenericConjugateGradientOptimizer::SetBetaDefinition( const BetaDefinitionType & arg )
{
  itkDebugMacro( "Setting BetaDefinition to " << arg );
  if( this->m_BetaDefinition != arg )
  {
    if( this->m_BetaDefinitionMap.count( arg ) != 1 )
    {
      itkExceptionMacro( << "Undefined beta: " << arg );
    }
    this->m_BetaDefinition = arg;
    this->Modified();
  }
}   // end SetBetaDefinition


/**
 * ******************** AddBetaDefinition ********************
 */

void
GenericConjugateGradientOptimizer::AddBetaDefinition(
  const BetaDefinitionType & name,
  ComputeBetaFunctionType function )
{
  itkDebugMacro( "Adding BetaDefinition: " << name );

  this->m_BetaDefinitionMap[ name ] = function;

}   // end AddBetaDefinition


/**
 * ********** SetMaxNrOfItWithoutImprovement *****************
 */

void
GenericConjugateGradientOptimizer::SetMaxNrOfItWithoutImprovement( unsigned long arg )
{
  itkDebugMacro( "Setting  to " << arg );
  this->m_UseDefaultMaxNrOfItWithoutImprovement = false;
  this->m_MaxNrOfItWithoutImprovement           = arg;
  this->Modified();
}   // end SetMaxNrOfItWithoutImprovement


/**
 * ********************* TestConvergence ************************
 */

bool
GenericConjugateGradientOptimizer::TestConvergence( bool itkNotUsed( firstLineSearchDone ) )
{
  itkDebugMacro( "TestConvergence" );

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


/**
 * ********************* PrintSelf ************************
 */

void
GenericConjugateGradientOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  /** Call the superclass' PrintSelf. */
  Superclass::PrintSelf( os, indent );

  os << indent << "m_CurrentGradient: "
     << this->m_CurrentGradient << std::endl;
  os << indent << "m_CurrentValue: "
     << this->m_CurrentValue << std::endl;
  os << indent << "m_CurrentIteration: "
     << this->m_CurrentIteration << std::endl;
  os << indent << "m_StopCondition: "
     << this->m_StopCondition << std::endl;
  os << indent << "m_Stop: "
     << ( this->m_Stop ? "true" : "false" ) << std::endl;
  os << indent << "m_CurrentStepLength: "
     << this->m_CurrentStepLength << std::endl;
  os << indent << "m_UseDefaultMaxNrOfItWithoutImprovement: "
     << ( this->m_UseDefaultMaxNrOfItWithoutImprovement ? "true" : "false" ) << std::endl;
  os << indent << "m_InLineSearch: "
     << ( this->m_InLineSearch ? "true" : "false" ) << std::endl;
  os << indent << "m_PreviousGradientAndSearchDirValid: "
     << ( this->m_PreviousGradientAndSearchDirValid ? "true" : "false" ) << std::endl;

  os << indent << "m_BetaDefinition: "
     << this->m_BetaDefinition << std::endl;
//   os << indent << "m_BetaDefinitionMap: "
//     << this->m_BetaDefinitionMap << std::endl;

  os << indent << "m_MaximumNumberOfIterations: "
     << this->m_MaximumNumberOfIterations << std::endl;
  os << indent << "m_ValueTolerance: "
     << this->m_ValueTolerance << std::endl;
  os << indent << "m_GradientMagnitudeTolerance: "
     << this->m_GradientMagnitudeTolerance << std::endl;
  os << indent << "m_MaxNrOfItWithoutImprovement: "
     << this->m_MaxNrOfItWithoutImprovement << std::endl;

  os << indent << "m_LineSearchOptimizer: "
     << this->m_LineSearchOptimizer.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // #ifndef __itkGenericConjugateGradientOptimizer_cxx
