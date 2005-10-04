#ifndef __elxRegularStepGradientDescent_h
#define __elxRegularStepGradientDescent_h

#include "itkImprovedRegularStepGradientDescentOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class RegularStepGradientDescent
	 * \brief An optimizer based on gradient descent...
	 *
	 * This optimizer is a wrap around the ImprovedRegularStepGradientDescentOptimizer,
	 * which is almost exactly equal to the itk::RegularStepGradientDescentOptimizer.
	 * This wrap-around class takes care of setting parameters, and printing progress
	 * information.
	 * For detailed information about the optimisation method, please read the
	 * documentation of the ImprovedRegularStepGradientDescentOptimizer class and the 
	 * itkRegularStepGradientDescentOptimizer (in the ITK-manual).
	 *
	 * The parameters used in this class are:
	 * \parameter MaximumNumberOfIterations: the maximum number of iterations in each resolution \n
	 *	 example: <tt>(MaximumNumberOfIterations 100 100 50)</tt> \n
	 *   Default value: 100.
	 * \parameter MinimumGradientMagnitude: stopping criterion. If the magnitude of the derivative
	 *   of the cost function is below this value, optimisation is stopped. \n
	 *	 example: <tt>(MinimumGradientMagnitude 0.0001 0.0001 0.001)</tt> \n
	 *   Default value: 1e-8. 
	 * \parameter MinimumStepLength: stopping criterion. If the steplength is below this
	 *   value, optimisation is stopped. \n
	 *	 example: <tt>(MinimumStepLength 1.0 0.5 0.1)</tt> \n
	 *   Default value: <em>0.5 / 2^resolutionlevel</em>
	 * \parameter MaximumStepLength: the starting steplength  \n
	 *	 example: <tt>(MaxiumStepLength 16.0 8.0 4.0)</tt> \n
	 *   Default value: <em>16 / 2^resolutionlevel</em> 
	 * \parameter RelaxationFactor: the factor with which the steplength is multiplied,
	 *   if the optimiser notices that a smaller steplength is needed. \n
	 *	 example: <tt>(RelaxationFactor 0.5 0.5 0.8 )</tt> \n
	 *   Default/recommended value: 0.5.
	 *	 
	 *
	 * RelaxationFactor
	 * 
	 * \sa ImprovedRegularStepGradientDescentOptimizer
	 * \ingroup Optimizers
	 */

	template <class TElastix>
		class RegularStepGradientDescent :
		public
			itk::ImprovedRegularStepGradientDescentOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef RegularStepGradientDescent									Self;
		typedef ImprovedRegularStepGradientDescentOptimizer	Superclass1;
		typedef OptimizerBase<TElastix>											Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( RegularStepGradientDescent, ImprovedRegularStepGradientDescentOptimizer );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific optimizer.
		 * example: <tt>(Optimizer "RegularStepGradientDescent")</tt>\n
		 */
		elxClassNameMacro( "RegularStepGradientDescent" );

		/** Typedef's inherited from Superclass1.*/
	  typedef Superclass1::CostFunctionType			CostFunctionType;
		typedef Superclass1::CostFunctionPointer	CostFunctionPointer;

		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;
		
		/** Typedef for the ParametersType. */
		typedef typename Superclass1::ParametersType				ParametersType;

		/** Methods invoked by elastix, in which parameters can be set and 
		 * progress information can be printed. */
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);		
		
		/** Override the SetInitialPosition.
		 * Override the implementation in itkOptimizer.h, to
		 * ensure that the scales array and the parameters
		 * array have the same size. */
		virtual void SetInitialPosition( const ParametersType & param );
		
	protected:

		  RegularStepGradientDescent();
			virtual ~RegularStepGradientDescent() {};
			
	private:

		  RegularStepGradientDescent( const Self& );	// purposely not implemented
			void operator=( const Self& );							// purposely not implemented
			
	}; // end class RegularStepGradientDescent
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxRegularStepGradientDescent.hxx"
#endif

#endif // end #ifndef __elxRegularStepGradientDescent_h
