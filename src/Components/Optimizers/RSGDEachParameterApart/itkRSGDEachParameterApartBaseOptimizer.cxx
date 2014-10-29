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

#ifndef __itkRSGDEachParameterApartBaseOptimizer_cxx
#define __itkRSGDEachParameterApartBaseOptimizer_cxx

#include "itkRSGDEachParameterApartBaseOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "vnl/vnl_math.h"

namespace itk
{

/**
 * Constructor
 */
RSGDEachParameterApartBaseOptimizer
::RSGDEachParameterApartBaseOptimizer()
{

  itkDebugMacro( "Constructor" );

  m_MaximumStepLength          = 1.0;
  m_MinimumStepLength          = 1e-3;
  m_GradientMagnitudeTolerance = 1e-4;
  m_NumberOfIterations         = 100;
  m_CurrentIteration           =   0;
  m_Value                      = 0;
  m_Maximize                   = false;
  m_CostFunction               = 0;

  m_CurrentStepLengths.Fill( 0.0f );
  m_CurrentStepLength = 0;

  m_StopCondition = MaximumNumberOfIterations;
  m_Gradient.Fill( 0.0f );
  m_PreviousGradient.Fill( 0.0f );

  m_GradientMagnitude = 0.0;

}


/**
 * Start the optimization
 */
void
RSGDEachParameterApartBaseOptimizer
::StartOptimization( void )
{

  itkDebugMacro( "StartOptimization" );

  m_CurrentIteration = 0;

  const unsigned int spaceDimension
    = m_CostFunction->GetNumberOfParameters();

  m_Gradient           = DerivativeType( spaceDimension );
  m_PreviousGradient   = DerivativeType( spaceDimension );
  m_CurrentStepLengths = DerivativeType( spaceDimension );
  m_Gradient.Fill( 0.0f );
  m_PreviousGradient.Fill( 0.0f );
  m_CurrentStepLengths.Fill( m_MaximumStepLength );
  m_CurrentStepLength = m_MaximumStepLength;

  this->SetCurrentPosition( GetInitialPosition() );
  this->ResumeOptimization();

}


/**
 * Resume the optimization
 */
void
RSGDEachParameterApartBaseOptimizer
::ResumeOptimization( void )
{

  itkDebugMacro( "ResumeOptimization" );

  m_Stop = false;

  this->InvokeEvent( StartEvent() );

  while( !m_Stop )
  {

    /** inefficient:
    ParametersType currentPosition = this->GetCurrentPosition();
    m_Value = m_CostFunction->GetValue( currentPosition );

    if( m_Stop )
      {
      break;
      }

    m_PreviousGradient = m_Gradient;
    m_CostFunction->GetDerivative( currentPosition, m_Gradient );
    */

    /** faster:*/

    m_PreviousGradient = m_Gradient;
    try
    {
      m_CostFunction->GetValueAndDerivative(
        this->GetCurrentPosition(), m_Value, m_Gradient );
    }
    catch( ExceptionObject & err )
    {
      m_StopCondition = MetricError;
      this->StopOptimization();
      throw err;
    }

    if( m_Stop )
    {
      break;
    }

    this->AdvanceOneStep();

    m_CurrentIteration++;

    if( m_CurrentIteration == m_NumberOfIterations )
    {
      m_StopCondition = MaximumNumberOfIterations;
      this->StopOptimization();
      break;
    }

  }

}


/**
 * Stop optimization
 */
void
RSGDEachParameterApartBaseOptimizer
::StopOptimization( void )
{

  itkDebugMacro( "StopOptimization" );

  m_Stop = true;
  this->InvokeEvent( EndEvent() );
}


/**
 * Advance one Step following the gradient direction
 */
void
RSGDEachParameterApartBaseOptimizer
::AdvanceOneStep( void )
{

  itkDebugMacro( "AdvanceOneStep" );

  const unsigned int spaceDimension
    = m_CostFunction->GetNumberOfParameters();

  DerivativeType transformedGradient( spaceDimension );
  DerivativeType previousTransformedGradient( spaceDimension );
  ScalesType     scales = this->GetScales();

  // Make sure the scales have been set properly
  if( scales.size() != spaceDimension )
  {
    itkExceptionMacro( << "The size of Scales is "
                       << scales.size()
                       << ", but the NumberOfParameters for the CostFunction is "
                       << spaceDimension
                       << "." );
  }

  for( unsigned int i = 0; i < spaceDimension; i++ )
  {
    transformedGradient[ i ] = m_Gradient[ i ] / scales[ i ];
    previousTransformedGradient[ i ]
      = m_PreviousGradient[ i ] / scales[ i ];
  }

  double magnitudeSquare = 0;
  for( unsigned int dim = 0; dim < spaceDimension; dim++ )
  {
    const double weighted = transformedGradient[ dim ];
    magnitudeSquare += weighted * weighted;
  }

  m_GradientMagnitude = vcl_sqrt( magnitudeSquare );

  if( m_GradientMagnitude < m_GradientMagnitudeTolerance )
  {
    m_StopCondition = GradientMagnitudeTolerance;
    StopOptimization();
    return;
  }

  double sumOfCurrentStepLengths  = 0.0;
  double biggestCurrentStepLength = 0.0;
  for( unsigned int i = 0; i < spaceDimension; i++ )
  {
    const bool signChange
      = ( transformedGradient[ i ] * previousTransformedGradient[ i ] ) < 0;

    if( signChange )
    {
      m_CurrentStepLengths[ i ] /= 2.0;
    }

    const double currentStepLengths_i = m_CurrentStepLengths[ i ];

    sumOfCurrentStepLengths += currentStepLengths_i;
    if( currentStepLengths_i > biggestCurrentStepLength )
    {
      biggestCurrentStepLength = currentStepLengths_i;
    }
  } //end for

  /** The average current step length: */
  m_CurrentStepLength = sumOfCurrentStepLengths / spaceDimension;

  /** if all current step lengths are smaller than the
   * MinimumStepLength stop the optimization
   */
  if( biggestCurrentStepLength < m_MinimumStepLength )
  {
    m_StopCondition = StepTooSmall;
    StopOptimization();
    return;
  }

  double direction;
  if( this->m_Maximize )
  {
    direction = 1.0;
  }
  else
  {
    direction = -1.0;
  }

  DerivativeType factor = DerivativeType( spaceDimension );

  for( unsigned int i = 0; i < spaceDimension; i++ )
  {
    factor[ i ] = direction * m_CurrentStepLengths[ i ] / m_GradientMagnitude;
  }

  // This method StepAlongGradient() will
  // be overloaded in non-vector spaces
  this->StepAlongGradient( factor, transformedGradient );

  this->InvokeEvent( IterationEvent() );

}


void
RSGDEachParameterApartBaseOptimizer
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "MaximumStepLength: "
     << m_MaximumStepLength << std::endl;
  os << indent << "MinimumStepLength: "
     << m_MinimumStepLength << std::endl;
  os << indent << "GradientMagnitudeTolerance: "
     << m_GradientMagnitudeTolerance << std::endl;
  os << indent << "NumberOfIterations: "
     << m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: "
     << m_CurrentIteration   << std::endl;
  os << indent << "Value: "
     << m_Value << std::endl;
  os << indent << "Maximize: "
     << m_Maximize << std::endl;
  if( m_CostFunction )
  {
    os << indent << "CostFunction: "
       << &m_CostFunction << std::endl;
  }
  else
  {
    os << indent << "CostFunction: "
       << "(None)" << std::endl;
  }
  os << indent << "CurrentStepLength: "
     << m_CurrentStepLength << std::endl;
  os << indent << "StopCondition: "
     << m_StopCondition << std::endl;
  os << indent << "Gradient: "
     << m_Gradient << std::endl;
}


} // end namespace itk

#endif
