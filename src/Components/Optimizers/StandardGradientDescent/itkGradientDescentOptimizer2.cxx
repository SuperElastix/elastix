#ifndef _itkGradientDescentOptimizer2_txx
#define _itkGradientDescentOptimizer2_txx

#include "itkGradientDescentOptimizer2.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

namespace itk
{

/**
 * Constructor
 */
GradientDescentOptimizer2
::GradientDescentOptimizer2()
{
  itkDebugMacro("Constructor");

  m_LearningRate = 1.0;
  m_NumberOfIterations = 100;
  m_CurrentIteration = 0;
  m_Value = 0.0;
  m_StopCondition = MaximumNumberOfIterations;
}



void
GradientDescentOptimizer2
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "LearningRate: "
     << m_LearningRate << std::endl;
  os << indent << "NumberOfIterations: "
     << m_NumberOfIterations << std::endl;
  os << indent << "CurrentIteration: "
     << m_CurrentIteration;
  os << indent << "Value: "
     << m_Value;
  os << indent << "StopCondition: "
     << m_StopCondition;
  os << std::endl;
  os << indent << "Gradient: "
     << m_Gradient;
  os << std::endl;
}


/**
 * Start the optimization
 */
void
GradientDescentOptimizer2
::StartOptimization( void )
{
  itkDebugMacro("StartOptimization");
   
  m_CurrentIteration   = 0;

  /** Get the number of parameters; checks also if a cost function has been set at all.
   * if not: an exception is thrown */
  const unsigned int numberOfParameters =
    this->GetScaledCostFunction()->GetNumberOfParameters();

  /** Initialize the scaledCostFunction with the currently set scales */
  this->InitializeScales();

  /** Set the current position as the scaled initial position */
  this->SetCurrentPosition( this->GetInitialPosition() );

  this->ResumeOptimization();
}


/**
 * Resume the optimization
 */
void
GradientDescentOptimizer2
::ResumeOptimization( void )
{
  
  itkDebugMacro("ResumeOptimization");

  m_Stop = false;

  InvokeEvent( StartEvent() );
  while( !m_Stop ) 
    {

    try
      {
      this->GetScaledValueAndDerivative( 
        this->GetScaledCurrentPosition(), m_Value, m_Gradient );
      }
    catch( ExceptionObject& err )
      {
      // An exception has occurred. 
      // Terminate immediately.
      m_StopCondition = MetricError;
      StopOptimization();

      // Pass exception to caller
      throw err;
      }


    if( m_Stop )
      {
      break;
      }
  
    AdvanceOneStep();

    m_CurrentIteration++;

    if( m_CurrentIteration >= m_NumberOfIterations )
      {
      m_StopCondition = MaximumNumberOfIterations;
      StopOptimization();
      break;
      }
    
    }
    

}


/**
 * Stop optimization
 */
void
GradientDescentOptimizer2
::StopOptimization( void )
{

  itkDebugMacro("StopOptimization");

  m_Stop = true;
  InvokeEvent( EndEvent() );
}





/**
 * Advance one Step following the gradient direction
 */
void
GradientDescentOptimizer2
::AdvanceOneStep( void )
{ 

  itkDebugMacro("AdvanceOneStep");

  const unsigned int spaceDimension = 
    this->GetScaledCostFunction()->GetNumberOfParameters();

  const ParametersType & currentPosition = this->GetScaledCurrentPosition();
 
  ParametersType newPosition( spaceDimension );
  for(unsigned int j = 0; j < spaceDimension; j++)
    {
    newPosition[j] = currentPosition[j] - m_LearningRate * this->m_Gradient[j];
    }

  this->SetScaledCurrentPosition( newPosition );

  this->InvokeEvent( IterationEvent() );

}



} // end namespace itk

#endif
