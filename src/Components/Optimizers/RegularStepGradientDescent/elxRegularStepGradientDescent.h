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
	 * This optimizer 
	 *
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
		
		/** Name of this class.*/
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
		
		/** Methods that have to be present everywhere.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);		
		
		/** Override the SetInitialPosition.*/
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
