#ifndef __itkImprovedRegularStepGradientDescentOptimizer_h
#define __itkImprovedRegularStepGradientDescentOptimizer_h

#include "itkRegularStepGradientDescentOptimizer.h"

namespace itk
{

		/**
	 * \class ImprovedRegularStepGradientDescentOptimizer
	 * \brief An optimizer based on gradient descent...
	 *
	 * This one is faster than the parent itk class: RegularStepGradientDescentOptimizer.
	 * Instead of GetValue and GetDerivative, the GetValueAndDerivative
	 * method is used, to obtain the metric value and its derivatives.
	 *
	 * \ingroup Optimizers
	 */

	class ImprovedRegularStepGradientDescentOptimizer :
		public RegularStepGradientDescentOptimizer
	{
	public:

		/** Standard ITK.*/
		typedef ImprovedRegularStepGradientDescentOptimizer		Self;
		typedef RegularStepGradientDescentOptimizer						Superclass;
		
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( ImprovedRegularStepGradientDescentOptimizer, RegularStepGradientDescentOptimizer );
		
		
		/** Typedef's inherited from Superclass.*/
	  typedef Superclass::CostFunctionType			CostFunctionType;
		typedef Superclass::CostFunctionPointer		CostFunctionPointer;
		typedef Superclass::StopConditionType			StopConditionType;
	
		 /** Start optimization. */
		void    StartOptimization( void );

		/** Resume previously stopped optimization with current parameters. */
		virtual void ImprovedResumeOptimization( void );

		virtual void StepAlongGradient(
			double factor, const DerivativeType & transformedGradient);

		itkGetConstMacro(GradientMagnitude, double);
		
	protected:

		  ImprovedRegularStepGradientDescentOptimizer()
			{
				m_GradientMagnitude = 0;
			}
			virtual ~ImprovedRegularStepGradientDescentOptimizer() {}
			
	private:

		  ImprovedRegularStepGradientDescentOptimizer( const Self& );	// purposely not implemented
			void operator=( const Self& );							// purposely not implemented
			
			double m_GradientMagnitude;

	}; // end class ImprovedRegularStepGradientDescentOptimizer
	

} // end namespace itk


#endif // end #ifndef __itkImprovedRegularStepGradientDescentOptimizer_h


