#ifndef __itkAcceleratedGradientDescentOptimizer_cxx
#define __itkAcceleratedGradientDescentOptimizer_cxx

#include "itkAcceleratedGradientDescentOptimizer.h"
#include "vnl/vnl_math.h"
#include "itkSigmoidImageFilter.h"

namespace itk
{

	/**
	 * ************************* Constructor ************************
	 */

	AcceleratedGradientDescentOptimizer
		::AcceleratedGradientDescentOptimizer()
	{
    this->m_UseCruzAcceleration = false;
    this->m_SigmoidMax = 1.0;
    this->m_SigmoidMin = -0.8;
    this->m_SigmoidScale = 1e-8;
				
	} // end Constructor

 
	/**
	 * ************************** UpdateCurrentTime ********************
	 *
	 * This function computes the input for the Compute_a function.
	 */

  void AcceleratedGradientDescentOptimizer
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
      this->Superclass::UpdateCurrentTime();
    }

  } // end UpdateCurrentTime
	

} // end namespace itk

#endif // end #ifndef __itkAcceleratedGradientDescentOptimizer_cxx

