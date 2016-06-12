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

#ifndef __itkFiniteDifferenceGradientDescentOptimizer_cxx
#define __itkFiniteDifferenceGradientDescentOptimizer_cxx

#include "itkFiniteDifferenceGradientDescentOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#include "math.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * ************************* Constructor ************************
 */

FiniteDifferenceGradientDescentOptimizer
::FiniteDifferenceGradientDescentOptimizer()
{
  itkDebugMacro( "Constructor" );

  this->m_Stop               = false;
  this->m_NumberOfIterations = 100;
  this->m_CurrentIteration   = 0;
  this->m_Value              = 0.0;
  this->m_StopCondition      = MaximumNumberOfIterations;

  this->m_GradientMagnitude   = 0.0;
  this->m_LearningRate        = 0.0;
  this->m_ComputeCurrentValue = false;
  this->m_Param_a             = 1.0;
  this->m_Param_c             = 1.0;
  this->m_Param_A             = 1.0;
  this->m_Param_alpha         = 0.602;
  this->m_Param_gamma         = 0.101;

}   // end Constructor


/**
 * ************************* PrintSelf **************************
 */

void
FiniteDifferenceGradientDescentOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  os << indent << "LearningRate: "
     << this->m_LearningRate << std::endl;
  os << indent << "NumberOfIterations: "
     << this->m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: "
     << this->m_CurrentIteration;
  os << indent << "Value: "
     << this->m_Value;
  os << indent << "StopCondition: "
     << this->m_StopCondition;
  os << std::endl;

}   // end PrintSelf


/**
 * *********************** StartOptimization ********************
 */
void
FiniteDifferenceGradientDescentOptimizer
::StartOptimization( void )
{
  itkDebugMacro( "StartOptimization" );

  this->m_CurrentIteration = 0;
  this->m_Stop             = false;

  /** Get the number of parameters; checks also if a cost function has been set at all.
   * if not: an exception is thrown */
  this->GetScaledCostFunction()->GetNumberOfParameters();

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
 * ********************** ResumeOptimization ********************
 */

void
FiniteDifferenceGradientDescentOptimizer
::ResumeOptimization( void )
{
  itkDebugMacro( "ResumeOptimization" );

  this->m_Stop = false;
  double       ck             = 1.0;
  unsigned int spaceDimension = 1;

  ParametersType param;
  double         valueplus;
  double         valuemin;

  InvokeEvent( StartEvent() );
  while( !this->m_Stop )
  {
    /** Get the Number of parameters.*/
    spaceDimension = this->GetScaledCostFunction()->GetNumberOfParameters();

    /** Initialisation.*/
    ck               = this->Compute_c( m_CurrentIteration );
    this->m_Gradient = DerivativeType( spaceDimension );
    param            = this->GetScaledCurrentPosition();

    /** Compute the current value, if desired by interested users */
    if( this->m_ComputeCurrentValue )
    {
      try
      {
        this->m_Value = this->GetScaledValue( param );
      }
      catch( ExceptionObject & err )
      {
        // An exception has occurred.
        // Terminate immediately.
        this->m_StopCondition = MetricError;
        StopOptimization();

        // Pass exception to caller
        throw err;
      }
      if( m_Stop )
      {
        break;
      }
    }   // if m_ComputeCurrentValue

    double sumOfSquaredGradients = 0.0;
    /** Calculate the derivative; this may take a while... */
    try
    {
      for( unsigned int j = 0; j < spaceDimension; j++ )
      {
        param[ j ] += ck;
        valueplus   = this->GetScaledValue( param );
        param[ j ] -= 2.0 * ck;
        valuemin    = this->GetScaledValue( param );
        param[ j ] += ck;

        const double gradient = ( valueplus - valuemin ) / ( 2.0 * ck );
        this->m_Gradient[ j ] = gradient;

        sumOfSquaredGradients += ( gradient * gradient );

      }   // for j = 0 .. spaceDimension
    }
    catch( ExceptionObject & err )
    {
      // An exception has occurred.
      // Terminate immediately.
      this->m_StopCondition = MetricError;
      StopOptimization();

      // Pass exception to caller
      throw err;
    }

    if( m_Stop )
    {
      break;
    }

    /** Save the gradient magnitude;
     * only for interested users... */
    this->m_GradientMagnitude = vcl_sqrt( sumOfSquaredGradients );

    this->AdvanceOneStep();

    this->m_CurrentIteration++;

    if( this->m_CurrentIteration >= this->m_NumberOfIterations )
    {
      this->m_StopCondition = MaximumNumberOfIterations;
      StopOptimization();
      break;
    }

  }   // while !m_stop

}   // end ResumeOptimization


/**
 * ********************** StopOptimization **********************
 */

void
FiniteDifferenceGradientDescentOptimizer
::StopOptimization( void )
{
  itkDebugMacro( "StopOptimization" );

  this->m_Stop = true;
  InvokeEvent( EndEvent() );

}   // end StopOptimization


/**
 * ********************** AdvanceOneStep ************************
 */

void
FiniteDifferenceGradientDescentOptimizer
::AdvanceOneStep( void )
{
  itkDebugMacro( "AdvanceOneStep" );

  const unsigned int spaceDimension
    = this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Compute the gain */
  double ak = this->Compute_a( this->m_CurrentIteration );

  /** Save it for users that are interested */
  this->m_LearningRate = ak;

  const ParametersType & currentPosition = this->GetScaledCurrentPosition();

  ParametersType newPosition( spaceDimension );
  for( unsigned int j = 0; j < spaceDimension; j++ )
  {
    newPosition[ j ] = currentPosition[ j ] - ak * this->m_Gradient[ j ];
  }

  this->SetScaledCurrentPosition( newPosition );

  this->InvokeEvent( IterationEvent() );

}   // end AdvanceOneStep


/**
 * ************************** Compute_a *************************
 *
 * This function computes the parameter a at iteration k, as
 * described by Spall.
 */

double
FiniteDifferenceGradientDescentOptimizer
::Compute_a( unsigned long k ) const
{
  return static_cast< double >(
    this->m_Param_a / vcl_pow( this->m_Param_A + k + 1, this->m_Param_alpha ) );

}   // end Compute_a


/**
 * ************************** Compute_c *************************
 *
 * This function computes the parameter a at iteration k, as
 * described by Spall.
 */

double
FiniteDifferenceGradientDescentOptimizer
::Compute_c( unsigned long k ) const
{
  return static_cast< double >(
    this->m_Param_c / vcl_pow( k + 1, this->m_Param_gamma ) );

}   // end Compute_c


} // end namespace itk

#endif // end #ifndef __itkFiniteDifferenceGradientDescentOptimizer_cxx
