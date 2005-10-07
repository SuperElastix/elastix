#ifndef __itkImprovedRegularStepGradientDescentOptimizer_cxx
#define __itkImprovedRegularStepGradientDescentOptimizer_cxx

#include "itkImprovedRegularStepGradientDescentOptimizer.h"
#include <math.h>

namespace itk
{

/**
 * Start the optimization
 */
void
ImprovedRegularStepGradientDescentOptimizer
::StartOptimization( void )
{
	/** This function is a copy of the version in
	 * itkRegularStepGradientDescentOptimizer, except
	 * for the last line
	 */

  itkDebugMacro("StartOptimization");

  m_CurrentStepLength         = m_MaximumStepLength;
  m_CurrentIteration          = 0;

  const unsigned int spaceDimension = 
    m_CostFunction->GetNumberOfParameters();

  m_Gradient = DerivativeType( spaceDimension );
  m_PreviousGradient = DerivativeType( spaceDimension );
  m_Gradient.Fill( 0.0f );
  m_PreviousGradient.Fill( 0.0f );

  this->SetCurrentPosition( GetInitialPosition() );
  this->ImprovedResumeOptimization();

}





/**
 * Resume the optimization
 */
void
ImprovedRegularStepGradientDescentOptimizer
::ImprovedResumeOptimization( void )
{
  
  itkDebugMacro("ResumeOptimization");

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
    catch( ExceptionObject& err)
    {
      //m_StopCondition = MetricError;
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
 * StepAlongGradient
 */
void
ImprovedRegularStepGradientDescentOptimizer
::StepAlongGradient(
	double factor, const DerivativeType & transformedGradient)
{
	/** Store the GradientMagnitude */
	m_GradientMagnitude = fabs( m_CurrentStepLength/factor );

	this->Superclass::StepAlongGradient(factor, transformedGradient);
}



} // end namespace itk

#endif // end #ifndef __itkImprovedRegularStepGradientDescentOptimizer_cxx

