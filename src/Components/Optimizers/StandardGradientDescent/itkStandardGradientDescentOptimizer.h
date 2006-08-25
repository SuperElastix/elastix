#ifndef __itkStandardGradientDescentOptimizer_h
#define __itkStandardGradientDescentOptimizer_h

#include "itkGradientDescentOptimizer2.h"

namespace itk
{

	/**
	 * \class StandardGradientDescentOptimizer
	 * \brief This class implements a gradient descent optimizer with a decaying gain.
	 *
	 * If \a C(x) is a costfunction that has to be minimised, the following iterative
	 * algorithm is used to find the optimal parameters \a x:
	 * 
	 * x(k+1) = x(k) - a(k) dC/dx
	 *
   * The gain \a a(k) at each iteration \a k is defined by:
	 *
	 * <em>a(k) =  a / (A + k + 1)^alpha</em>.
	 * 
	 * It is very suitable to be used in combination with a stochastic estimate
	 * of the gradient \a dC/dx. For example, in image registration problems it is
	 * often advantageous to compute the metric derivative (\a dC/dx) on a new set 
	 * of randomly selected image samples in each iteration. You may set the parameter
	 * \c NewSamplesEveryIteration to \c "true" to achieve this effect.
	 * For more information on this strategy, you may have a look at:
	 *
	 * S. Klein, M. Staring, J.P.W. Pluim,
	 * "Comparison of gradient approximation techniques for optimisation of mutual information in nonrigid registration",
	 * in: SPIE Medical Imaging: Image Processing,
	 * Editor(s): J.M. Fitzpatrick, J.M. Reinhardt, SPIE press, 2005, vol. 5747, Proceedings of SPIE, pp. 192-203.
	 *   
	 * Note that the gain is computed in the same way as in the
	 * SimultaneousPerturbationOptimizer
	 * 
	 * \sa SimultaneousPerturbationOptimizer, StandardGradientDescent
	 * \ingroup Optimizers
	 */

	class StandardGradientDescentOptimizer :
		public GradientDescentOptimizer2
	{
	public:

		/** Standard ITK.*/
		typedef StandardGradientDescentOptimizer		Self;
		typedef GradientDescentOptimizer2						Superclass;
		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( StandardGradientDescentOptimizer, GradientDescentOptimizer2 );
				
		/** Typedef's inherited from Superclass.*/
	  typedef Superclass::CostFunctionType			CostFunctionType;
		typedef Superclass::CostFunctionPointer		CostFunctionPointer;
		typedef Superclass::StopConditionType			StopConditionType;
	
		/** Set/Get a. */
		itkSetMacro( Param_a, double );
		itkGetConstMacro( Param_a, double );
		
		/** Set/Get A. */
		itkSetMacro( Param_A, double );
		itkGetConstMacro( Param_A, double );
		
		/** Set/Get alpha. */
		itkSetMacro( Param_alpha, double );
		itkGetConstMacro( Param_alpha, double );
		
		/** 
		 * Sets a new LearningRate before calling the Superclass'
		 * implementation
		 */
		virtual void AdvanceOneStep(void);

		/** Compute the gradient magnitude */
		const double GetGradientMagnitude(void) const;
		
	protected:

		  StandardGradientDescentOptimizer();
			virtual ~StandardGradientDescentOptimizer() {};

		/** Function to compute the parameter at iteration k. */
		virtual double Compute_a( unsigned long k ) const;

			
	private:

		  StandardGradientDescentOptimizer( const Self& );	// purposely not implemented
			void operator=( const Self& );							// purposely not implemented

		/**Parameters, as described by Spall.*/
		double												m_Param_a;
		double												m_Param_A;
		double												m_Param_alpha;
		
			
	}; // end class StandardGradientDescentOptimizer
	

} // end namespace itk


#endif // end #ifndef __itkStandardGradientDescentOptimizer_h


