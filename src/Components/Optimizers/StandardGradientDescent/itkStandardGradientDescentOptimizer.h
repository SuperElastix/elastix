#ifndef __itkStandardGradientDescentOptimizer_h
#define __itkStandardGradientDescentOptimizer_h

#include "itkGradientDescentOptimizer.h"

namespace itk
{

	/**
	 * *********** StandardGradientDescent ******************
	 *
	 * The StandardGradientDescent class.
	 *
	 * It adds calculation of the steplength in each iteration,
	 * in a way similar to the Simultaneous Perturbation
	 * 
	 */

	class StandardGradientDescentOptimizer :
		public GradientDescentOptimizer
	{
	public:

		/** Standard ITK.*/
		typedef StandardGradientDescentOptimizer		Self;
		typedef GradientDescentOptimizer						Superclass;
		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( StandardGradientDescentOptimizer, GradientDescentOptimizer );
		
		
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

		const double GetGradientMagnitude(void) const;
		
	protected:

		  StandardGradientDescentOptimizer();
			virtual ~StandardGradientDescentOptimizer() {};

		// Function to compute the parameter at iteration k.
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


