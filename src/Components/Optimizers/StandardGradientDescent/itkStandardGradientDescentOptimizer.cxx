#ifndef __itkStandardGradientDescentOptimizer_cxx
#define __itkStandardGradientDescentOptimizer_cxx

#include "itkStandardGradientDescentOptimizer.h"
#include "math.h"
#include "vnl/vnl_math.h"

namespace itk
{

	/**
	 * ************************* Constructor ************************
	 */

	StandardGradientDescentOptimizer
		::StandardGradientDescentOptimizer()
	{
		m_Param_a = 1.0;
		m_Param_A = 1.0;
		m_Param_alpha = 0.602;
		
		
	} // end Constructor
	

	/**
	 * ******************** AdvanceOneStep **************************
	 */

	void StandardGradientDescentOptimizer::AdvanceOneStep(void)
	{
		
		this->SetLearningRate(
			this->Compute_a( this->GetCurrentIteration() )		);

		this->Superclass::AdvanceOneStep();

	}


	/**
	 * ************************** Compute_a *************************
	 *
	 * This function computes the parameter a at iteration k, as
	 * described by Spall.
	 */

	double StandardGradientDescentOptimizer
		::Compute_a(unsigned long k) const
	{ 
		
		return static_cast<double>(
			m_Param_a / pow( m_Param_A + k + 1, m_Param_alpha ) );

	} // end Compute_a
	

	const double StandardGradientDescentOptimizer
		::GetGradientMagnitude(void) const
	{
		const unsigned int spaceDimension =
			this->GetScaledCostFunction()->GetNumberOfParameters();

		double squared_gradient_sum = 0;
		for (unsigned int j = 0; j < spaceDimension; j++)
		{
			const double grad = m_Gradient[j];
			squared_gradient_sum += grad * grad;
		}
		
		return vcl_sqrt(squared_gradient_sum);
	
	}




} // end namespace itk

#endif // end #ifndef __itkStandardGradientDescentOptimizer_cxx

