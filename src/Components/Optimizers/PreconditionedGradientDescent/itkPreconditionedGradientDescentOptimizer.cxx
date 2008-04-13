#ifndef _itkPreconditionedGradientDescentOptimizer_txx
#define _itkPreconditionedGradientDescentOptimizer_txx

#include "itkPreconditionedGradientDescentOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

namespace itk
{

  /**
  * ****************** Constructor ************************
  */

  PreconditionedGradientDescentOptimizer
    ::PreconditionedGradientDescentOptimizer()
  {
    itkDebugMacro("Constructor");

    this->m_LearningRate = 1.0;
    this->m_NumberOfIterations = 100;
    this->m_CurrentIteration = 0;
    this->m_Value = 0.0;
    this->m_StopCondition = MaximumNumberOfIterations;
    this->m_MinimumConditionNumber = 0.01;
    this->m_EigenSystem = 0;

  } // end Constructor

  /**
  * ****************** Destructor ************************
  */

  PreconditionedGradientDescentOptimizer
    ::~PreconditionedGradientDescentOptimizer()
  {
    itkDebugMacro("Destructor");

    if ( this->m_EigenSystem )
    {
      delete this->m_EigenSystem;
      this->m_EigenSystem = 0;
    }    

  } // end Destructor



  /** 
   * *************** PrintSelf *************************
   */

  void
    PreconditionedGradientDescentOptimizer
    ::PrintSelf(std::ostream& os, Indent indent) const
  {
    this->Superclass::PrintSelf(os,indent);

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
    os << indent << "Gradient: "
      << this->m_Gradient;
    os << std::endl;
  } // end PrintSelf


  /**
  * **************** Start the optimization ********************
  */

  void
    PreconditionedGradientDescentOptimizer
    ::StartOptimization( void )
  {
    itkDebugMacro("StartOptimization");

    this->m_CurrentIteration   = 0;

    /** Get the number of parameters; checks also if a cost function has been set at all.
    * if not: an exception is thrown */
    const unsigned int numberOfParameters =
      this->GetScaledCostFunction()->GetNumberOfParameters();

    /** Initialize the scaledCostFunction with the currently set scales */
    this->InitializeScales();

    /** Set the current position as the scaled initial position */
    this->SetCurrentPosition( this->GetInitialPosition() );

    this->ResumeOptimization();
  } // end StartOptimization


  /**
  * ************************ ResumeOptimization *************
  */

  void
    PreconditionedGradientDescentOptimizer
    ::ResumeOptimization( void )
  {
    itkDebugMacro("ResumeOptimization");

    this->m_Stop = false;

    InvokeEvent( StartEvent() );
    while( ! this->m_Stop ) 
    {

      try
      {
        this->GetScaledValueAndDerivative( 
          this->GetScaledCurrentPosition(), m_Value, m_Gradient );
      }
      catch ( ExceptionObject& err )
      {
        this->MetricErrorResponse( err );
      }

      /** StopOptimization may have been called. */
      if ( this->m_Stop )
      {
        break;
      }

      this->AdvanceOneStep();

      /** StopOptimization may have been called. */
      if ( this->m_Stop )
      {
        break;
      }

      this->m_CurrentIteration++;

      if ( m_CurrentIteration >= m_NumberOfIterations )
      {
        this->m_StopCondition = MaximumNumberOfIterations;
        this->StopOptimization();
        break;
      }

    } // end while

  } // end ResumeOptimization()


 /**
  * ***************** MetricErrorResponse ************************
  */

  void
    PreconditionedGradientDescentOptimizer
    ::MetricErrorResponse( ExceptionObject & err )
  {
    /** An exception has occurred. Terminate immediately. */
    this->m_StopCondition = MetricError;
    this->StopOptimization();

    /** Pass exception to caller. */
    throw err;

  } // end MetricErrorResponse()


  /**
  * ***************** Stop optimization ************************
  */

  void
    PreconditionedGradientDescentOptimizer
    ::StopOptimization( void )
  {
    itkDebugMacro("StopOptimization");

    this->m_Stop = true;
    this->InvokeEvent( EndEvent() );
  } // end StopOptimization


  /**
  * ************ AdvanceOneStep ****************************
  * following the gradient direction
  */

  void
    PreconditionedGradientDescentOptimizer
    ::AdvanceOneStep( void )
  { 
    itkDebugMacro("AdvanceOneStep");

    const unsigned int spaceDimension = 
      this->GetScaledCostFunction()->GetNumberOfParameters();

    const ParametersType & currentPosition = this->GetScaledCurrentPosition();

    vnl_vector<PreconditionValueType> searchDirection;
    if ( ! this->m_EigenSystem )
    {
      searchDirection = this->m_Gradient;
    }
    else
    {
      searchDirection = 
        this->m_EigenSystem->V * ( this->m_EigenSystem->D * this->m_Gradient.post_multiply(
        this->m_EigenSystem->V ) );
    }

    ParametersType newPosition( spaceDimension );
    for(unsigned int j = 0; j < spaceDimension; j++)
    {
      newPosition[j] = currentPosition[j] - this->m_LearningRate * searchDirection[j];
    }

    this->SetScaledCurrentPosition( newPosition );

    this->InvokeEvent( IterationEvent() );

  } // end AdvanceOneStep


 /**
  * ************ SetPreconditionMatrix ****************************
  * Compute and modify eigensystem of preconditioning matrix.
  * Does not take into account scales (yet)!
  */

  void
    PreconditionedGradientDescentOptimizer
    ::SetPreconditionMatrix( const PreconditionType & precondition )
  { 
    itkDebugMacro("SetPreconditionMatrix");
   
    const unsigned int spaceDimension = precondition.cols();

    /** Compute eigen system */
    if ( this->m_EigenSystem )
    {
      delete this->m_EigenSystem;
      this->m_EigenSystem = 0;
    }
    this->m_EigenSystem = new EigenSystemType( precondition );

    /** Max eigenvalue measured and minimum eigenvalue allowed */
    const double maxeig = this->m_EigenSystem->D[ spaceDimension-1  ];
    const double mineig = maxeig * this->GetMinimumConditionNumber();
    
    /** The eigen vector with the lowest valid eigen value  */
    unsigned int lowestValidEigVec = spaceDimension-1;

    /** Invert eigenvalues if >= mineig and >0 and compute lowest valid eigen vector */
    for (unsigned int i = spaceDimension; i > 0; --i )
    {
      const unsigned int eignr = i-1;      
      double eigval = this->m_EigenSystem->D( eignr );
      
      if ( eigval < mineig )
      {
        eigval = mineig;        
      }
      if (eigval > 1e-14 )
      {
        eigval = 1.0 / eigval;
        lowestValidEigVec = eignr;
      }
      this->m_EigenSystem->D( eignr ) = eigval;
    }
    unsigned int numberOfValidEigVecs = spaceDimension - lowestValidEigVec;
  
    /** Extract part of eigenvalue matrix that is >0 */
    this->m_EigenSystem->D = this->m_EigenSystem->D.diagonal().extract(
      numberOfValidEigVecs, lowestValidEigVec );
    
    /** Extract corresponding part of eigenvector matrix */
    this->m_EigenSystem->V = this->m_EigenSystem->V.extract( 
      spaceDimension, numberOfValidEigVecs, 0, lowestValidEigVec );

  } // end SetPreconditionMatrix


} // end namespace itk

#endif
