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
		
		m_NumberOfIterations = 100;
		m_CurrentIteration = 0;
		m_Maximize = false;
		m_Value = 0.0;
		m_StopCondition = MaximumNumberOfIterations;
		
		m_GradientMagnitude = 0.0;
		m_LearningRate = 0.0;
		m_ComputeCurrentValue = false;
		m_Param_a = 1.0;
		m_Param_c = 1.0;
		m_Param_A = 1.0;
		m_Param_alpha = 0.602;
		m_Param_gamma = 0.101;		
		
	} // end Constructor
	
	
	/**
	 * ************************* PrintSelf **************************
	 */

	void
		FiniteDifferenceGradientDescentOptimizer
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		Superclass::PrintSelf( os, indent );
		
		os << indent << "LearningRate: "
		   << m_LearningRate << std::endl;
		os << indent << "NunberOfIterations: "
			<< m_NumberOfIterations << std::endl;
		os << indent << "Maximize: "
			<< m_Maximize << std::endl;
		os << indent << "CurrentIteration: "
			<< m_CurrentIteration;
		os << indent << "Value: "
			<< m_Value;
		if ( m_CostFunction )
    {
			os << indent << "CostFunction: "
				<< m_CostFunction;
    }
		os << indent << "StopCondition: "
			<< m_StopCondition;
		os << std::endl;
		
	} // end PrintSelf
	
	
	/**
	 * *********************** StartOptimization ********************
	 */
	void
		FiniteDifferenceGradientDescentOptimizer
		::StartOptimization(void)
	{		
		itkDebugMacro( "StartOptimization" );
		
		m_CurrentIteration = 0;
		
		this->SetCurrentPosition( this->GetInitialPosition() );
		this->ResumeOptimization();
		
	} // end StartOptimization	
	
	
	/**
	 * ********************** ResumeOptimization ********************
	 */

	void
		FiniteDifferenceGradientDescentOptimizer
		::ResumeOptimization( void )
	{		
		itkDebugMacro( "ResumeOptimization" );
		
		m_Stop = false;
		double ck = 1.0;
		unsigned int spaceDimension = 1;
		
		ParametersType param;
		double valueplus;
		double valuemin;
		
		InvokeEvent( StartEvent() );
		while( !m_Stop ) 
		{
			/** Get the Number of parameters.*/
			spaceDimension = m_CostFunction->GetNumberOfParameters();
			
			/** Initialisation.*/
			ck					= this->Compute_c( m_CurrentIteration );
			m_Gradient	=	DerivativeType( spaceDimension );
			param = this->GetCurrentPosition();
		
			/** Compute the current value, if desired by interested users */
			if ( m_ComputeCurrentValue )
			{
				try
				{
					m_Value = m_CostFunction->GetValue( param );
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
			} // if m_ComputeCurrentValue
			
		
			double sumOfSquaredGradients = 0.0;
			/** Calculate the derivative; this may take a while... */
			try 
			{
				for ( unsigned int j = 0; j < spaceDimension; j++ )
				{
					param[j] += ck;
					valueplus = m_CostFunction->GetValue( param );
					param[j] -= 2.0*ck;
					valuemin = m_CostFunction->GetValue( param );
					param[j] += ck;
								
					const double gradient = (valueplus - valuemin) / (2.0 * ck);
					m_Gradient[j] = gradient;
					
					sumOfSquaredGradients += ( gradient * gradient );
					
				} // for j = 0 .. spaceDimension
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
			
			/** Save the gradient magnitude; 
			 * only for interested users... */
			m_GradientMagnitude = vcl_sqrt( sumOfSquaredGradients );
			
			AdvanceOneStep();
			
			m_CurrentIteration++;
			
			if( m_CurrentIteration >= m_NumberOfIterations )
			{
				m_StopCondition = MaximumNumberOfIterations;
				StopOptimization();
				break;
			}
			
		} // while !m_stop
    
		
	} // end ResumeOptimization
	
	
	/**
	 * ********************** StopOptimization **********************
	 */

	void
		FiniteDifferenceGradientDescentOptimizer
		::StopOptimization( void )
	{		
		itkDebugMacro( "StopOptimization" );
		
		m_Stop = true;
		InvokeEvent( EndEvent() );

	} // end StopOptimization
	
	
	/**
	 * ********************** AdvanceOneStep ************************
	 */

	void
		FiniteDifferenceGradientDescentOptimizer
		::AdvanceOneStep( void )
	{		
		itkDebugMacro( "AdvanceOneStep" );
		
		double direction;
		if( this->m_Maximize ) direction = 1.0;
		else direction = -1.0;
		
		const unsigned int spaceDimension = 
			m_CostFunction->GetNumberOfParameters();
		
		/** Compute the gain */
		double ak = this->Compute_a( m_CurrentIteration );

		/** Save it for users that are interested */
		m_LearningRate = ak;
		
		const ParametersType & currentPosition = this->GetCurrentPosition();
		
		ScalesType scales = this->GetScales();
		
		DerivativeType transformedGradient( spaceDimension ); 
		
		for ( unsigned int j = 0; j < spaceDimension; j++ )
    {
			transformedGradient[ j ] = m_Gradient[ j ] / scales[ j ];
    }
		
		ParametersType newPosition( spaceDimension );
		for ( unsigned int j = 0; j < spaceDimension; j++ )
    {
			newPosition[ j ] = currentPosition[ j ] + 
				direction * ak * transformedGradient[ j ];
    }
		
		this->SetCurrentPosition( newPosition );
		
		this->InvokeEvent( IterationEvent() );
		
	} // end AdvanceOneStep
	
	
	
	
	/**
	 * ************************** Compute_a *************************
	 *
	 * This function computes the parameter a at iteration k, as
	 * described by Spall.
	 */

	double FiniteDifferenceGradientDescentOptimizer
		::Compute_a( unsigned long k ) const
	{ 
		return static_cast<double>(
			m_Param_a / pow( m_Param_A + k + 1, m_Param_alpha ) );
		
	} // end Compute_a
	
	
	
	/**
	 * ************************** Compute_c *************************
	 *
	 * This function computes the parameter a at iteration k, as
	 * described by Spall.
	 */
	
	double FiniteDifferenceGradientDescentOptimizer
		::Compute_c( unsigned long k ) const
	{ 
		return static_cast<double>(
			m_Param_c / pow( k + 1, m_Param_gamma ) );
		
	} // end Compute_c

	
} // end namespace itk


#endif // end #ifndef __itkFiniteDifferenceGradientDescentOptimizer_cxx

