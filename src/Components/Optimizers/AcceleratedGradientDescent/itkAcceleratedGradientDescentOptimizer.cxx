#ifndef __itkStandardGradientDescentOptimizer_cxx
#define __itkStandardGradientDescentOptimizer_cxx

#include "itkStandardGradientDescentOptimizer.h"
#include "math.h"
#include "vnl/vnl_math.h"
#include "itkSigmoidImageFilter.h"

namespace itk
{

	/**
	 * ************************* Constructor ************************
	 */

	StandardGradientDescentOptimizer
		::StandardGradientDescentOptimizer()
	{
		this->m_Param_a = 1.0;
		this->m_Param_A = 1.0;
		this->m_Param_alpha = 0.602;

    this->m_UseCruzAcceleration = false;
    this->m_SigmoidMax = 1.0;
    this->m_SigmoidMin = -0.999;
    this->m_SigmoidScale = 1e-8;
    this->m_CurrentTime = 0.0;
    this->m_InitialTime = 10.0;
				
	} // end Constructor


  /**
   * ********************** StartOptimization *********************
   */

  void StandardGradientDescentOptimizer::StartOptimization(void)
  {
    this->m_CurrentTime = this->m_InitialTime;
    this->Superclass::StartOptimization();
  } // end StartOptimization


	/**
	 * ******************** AdvanceOneStep **************************
	 */

	void StandardGradientDescentOptimizer::AdvanceOneStep(void)
	{
		
		this->SetLearningRate( this->Compute_a( this->m_CurrentTime ) );

		this->Superclass::AdvanceOneStep();

    this->UpdateCurrentTime();

	} // end AdvanceOneStep


	/**
	 * ************************** Compute_a *************************
	 *
	 * This function computes the parameter a at iteration/time k, as
	 * described by Spall.
	 */

	double StandardGradientDescentOptimizer
		::Compute_a(double k) const
	{ 
		return static_cast<double>(
			m_Param_a / pow( m_Param_A + k + 1.0, m_Param_alpha ) );

	} // end Compute_a

  
	/**
	 * ************************** UpdateCurrentTime ********************
	 *
	 * This function computes the input for the Compute_a function.
	 */

  void StandardGradientDescentOptimizer
		::UpdateCurrentTime( void )
  {
    typedef itk::Function::Sigmoid<double, double> SigmoidType;

    if ( this->m_UseCruzAcceleration )
    {
      if ( this->GetCurrentIteration() > 0 )
      {
        /** Make sigmoid function 
         * Compute beta such that sigmoid(0)=0 
         * We assume Max>0, min<0 */
        SigmoidType sigmoid;
        sigmoid.SetOutputMaximum( this->GetSigmoidMax() );
        sigmoid.SetOutputMinimum( this->GetSigmoidMin() );
        sigmoid.SetAlpha( this->GetSigmoidScale() );
        const double beta = this->GetSigmoidScale() *
           vcl_log( - this->GetSigmoidMax() / this->GetSigmoidMin() );
        sigmoid.SetBeta( beta );

        /** Formula (2) in Cruz */
        const double inprod = inner_product(
          this->m_PreviousGradient, this->GetGradient() );
        this->m_CurrentTime = this->m_CurrentTime + sigmoid(-inprod);
        this->m_CurrentTime = vnl_math_max( 0.0, this->m_CurrentTime );        
      }

      /** Save for next iteration */
      this->m_PreviousGradient = this->GetGradient();
    }
    else
    {
      /** Simply Robbins-Monro: time=iterationnr. */
      this->m_CurrentTime += 1.0;
    }

  } // end UpdateCurrentTime
	

} // end namespace itk

#endif // end #ifndef __itkStandardGradientDescentOptimizer_cxx

