#ifndef __elxFiniteDifferenceGradientDescent_h
#define __elxFiniteDifferenceGradientDescent_h

#include "itkFiniteDifferenceGradientDescentOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;

	/**
	 * \class FiniteDifferenceGradientDescent
	 * \brief An optimizer based on gradient descent ...
	 *
	 * This optimizer ...
	 *
	 * \ingroup Optimizers
	 */

	template <class TElastix>
		class FiniteDifferenceGradientDescent :
		public
			itk::FiniteDifferenceGradientDescentOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef FiniteDifferenceGradientDescent						Self;
		typedef FiniteDifferenceGradientDescentOptimizer	Superclass1;
		typedef OptimizerBase<TElastix>										Superclass2;
		typedef SmartPointer<Self>												Pointer;
		typedef SmartPointer<const Self>									ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( FiniteDifferenceGradientDescent, FiniteDifferenceGradientDescentOptimizer );
		
		/** Name of this class.*/
		elxClassNameMacro( "FiniteDifferenceGradientDescent" );

		/** Typedef's inherited from Superclass1.*/
	  typedef Superclass1::CostFunctionType			CostFunctionType;
		typedef Superclass1::CostFunctionPointer	CostFunctionPointer;
		typedef Superclass1::StopConditionType		StopConditionType;
		
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

		/** Methods that have to be present everywhere.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);		
		
		/** Override the SetInitialPosition.*/
		virtual void SetInitialPosition( const ParametersType & param );
		
	protected:

		  FiniteDifferenceGradientDescent();
			virtual ~FiniteDifferenceGradientDescent() {};
			
			bool m_ShowMetricValues;
			
	private:

		  FiniteDifferenceGradientDescent( const Self& );	// purposely not implemented
			void operator=( const Self& );							// purposely not implemented
			
	}; // end class FiniteDifferenceGradientDescent
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxFiniteDifferenceGradientDescent.hxx"
#endif

#endif // end #ifndef __elxFiniteDifferenceGradientDescent_h

