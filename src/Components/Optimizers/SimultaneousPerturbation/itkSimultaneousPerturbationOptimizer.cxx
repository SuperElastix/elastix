#ifndef __itkSimultaneousPerturbationOptimizer_cxx
#define __itkSimultaneousPerturbationOptimizer_cxx

#include "itkSimultaneousPerturbationOptimizer.h"
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#include "math.h"
#include "elx_sample.h"
#include "vnl/vnl_math.h"


namespace itk
{


	/**
	 * ************************* Constructor ************************
	 */

	SimultaneousPerturbationOptimizer
		::SimultaneousPerturbationOptimizer()
	{
		itkDebugMacro( "Constructor" );
		
		m_NumberOfIterations = 100;
		m_CurrentIteration = 0;
		m_Maximize = false;
		m_Value = 0.0;
		m_StopCondition = MaximumNumberOfIterations;
		
		m_CurrentStepLength = 0.0;
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
		SimultaneousPerturbationOptimizer
		::PrintSelf( std::ostream& os, Indent indent ) const
	{
		Superclass::PrintSelf( os, indent );
		
		//os << indent << "LearningRate: "
		//   << m_LearningRate << std::endl;
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
		SimultaneousPerturbationOptimizer
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
		SimultaneousPerturbationOptimizer
		::ResumeOptimization( void )
	{		
		itkDebugMacro( "ResumeOptimization" );
		
		m_Stop = false;
		double ck = 1.0;
		unsigned int spaceDimension = 1;
		
		ParametersType phiplus;
		ParametersType phimin;
		double valueplus;
		double valuemin;
		double valuediff;
		
		InvokeEvent( StartEvent() );
		while( !m_Stop ) 
		{
			/** Get the Number of parameters.*/
			spaceDimension = m_CostFunction->GetNumberOfParameters();
			
			/** Initialisation.*/
			this->GenerateDelta( spaceDimension );
			ck					= this->Compute_c( m_CurrentIteration );
			m_Gradient	=	DerivativeType( spaceDimension );
			phiplus			=	DerivativeType( spaceDimension );
			phimin			=	DerivativeType( spaceDimension );
			const ParametersType & phi = this->GetCurrentPosition();
			for ( unsigned int j = 0; j < spaceDimension; j++ )
			{
				phiplus[ j ]	= phi[ j ] + ck * m_Delta[ j ];
				phimin[ j ]		= phi[ j ] - ck * m_Delta[ j ];
			}
			
			try
			{
				valueplus = m_CostFunction->GetValue( phiplus );
				valuemin = m_CostFunction->GetValue( phimin );
				if ( m_ComputeCurrentValue )
				{
					m_Value = m_CostFunction->GetValue( phi );
				}
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
			
			valuediff = ( valueplus - valuemin ) / ( 2 * ck );
			
			
			for ( unsigned int j = 0; j < spaceDimension; j++ )
			{
				m_Gradient[ j ] = valuediff / m_Delta[ j ];
			}
			
			/** Save the gradient magnitude; only valid if the elements of 
			 * m_Delta have a constant magnitude of 1, which is the case for SP;
			 * only for interested users... */
			m_GradientMagnitude = static_cast<double>
							(valuediff * valuediff * spaceDimension);
			
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
		SimultaneousPerturbationOptimizer
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
		SimultaneousPerturbationOptimizer
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
		
		/** The step length is defined as ABS(ak * gradient);
		 * for interested users... */
		m_CurrentStepLength = fabs( ak * m_Gradient[ 0 ] * m_Delta[ 0 ] );
		
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

	double SimultaneousPerturbationOptimizer
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
	
	double SimultaneousPerturbationOptimizer
		::Compute_c( unsigned long k ) const
	{ 
		return static_cast<double>(
			m_Param_c / pow( k + 1, m_Param_gamma ) );
		
	} // end Compute_c
	
	
	
	/**
	 * ********************** GenerateDelta *************************
	 *
	 * This function generates a perturbation vector delta.
	 * Currently the elements are drawn from a bernouilli
	 * distribution. (+- 1)
	 */

	void SimultaneousPerturbationOptimizer
		::GenerateDelta( const unsigned int spaceDimension )
	{ 
		m_Delta = DerivativeType( spaceDimension );
		
		for ( unsigned int j = 0; j < spaceDimension; j++ )
		{
			m_Delta[ j ] = 2 * vnl_math_rnd( elx_sample_uniform(0,1) ) - 1;
		}

	} // end GenerateDelta
	

	
} // end namespace itk


#endif // end #ifndef __itkSimultaneousPerturbationOptimizer_cxx

