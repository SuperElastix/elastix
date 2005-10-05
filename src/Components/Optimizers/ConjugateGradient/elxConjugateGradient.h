#ifndef __elxConjugateGradient_h
#define __elxConjugateGradient_h

#include "itkGenericConjugateGradientOptimizer.h"
#include "itkMoreThuenteLineSearchOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;


	/**
	 * \class ConjugateGradient
	 * \brief An optimizer based on the itk::GenericConjugateGradientOptimizer.
	 *
	 * The ConjugateGradient class ....
	 *
	 * The parameters used in this class are:
	 * \parameter Optimizer: Select this optimizer as follows:\n
	 *		<tt>(Optimizer "ConjugateGradient")</tt>
	 * \parameter alotofothers: ???
	 *
	 * \ingroup Optimizers
	 */

	template <class TElastix>
		class ConjugateGradient :
		public
			itk::GenericConjugateGradientOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef ConjugateGradient						  			Self;
		typedef GenericConjugateGradientOptimizer						Superclass1;
		typedef OptimizerBase<TElastix>							Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( ConjugateGradient, GenericConjugateGradientOptimizer );
		
		/** Name of this class.
		 * Use this name in the parameter file to select this specific optimizer. \n
		 * example: <tt>(Optimizer "ConjugateGradient")</tt>\n
		 */
		elxClassNameMacro( "ConjugateGradient" );

		/** Typedef's inherited from Superclass1.*/
	  typedef Superclass1::CostFunctionType								    CostFunctionType;
		typedef Superclass1::CostFunctionPointer						    CostFunctionPointer;
		typedef Superclass1::StopConditionType							    StopConditionType;
		typedef Superclass1::ParametersType									    ParametersType;
		typedef Superclass1::DerivativeType											DerivativeType; 
		typedef Superclass1::ScalesType													ScalesType; 
		
		/** Typedef's inherited from Elastix.*/
		typedef typename Superclass2::ElastixType						ElastixType;
		typedef typename Superclass2::ElastixPointer				ElastixPointer;
		typedef typename Superclass2::ConfigurationType			ConfigurationType;
		typedef typename Superclass2::ConfigurationPointer	ConfigurationPointer;
		typedef typename Superclass2::RegistrationType			RegistrationType;
		typedef typename Superclass2::RegistrationPointer		RegistrationPointer;
		typedef typename Superclass2::ITKBaseType						ITKBaseType;

		/** Extra typedefs */
		typedef MoreThuenteLineSearchOptimizer							LineOptimizerType;
		typedef LineOptimizerType::Pointer									LineOptimizerPointer;
		typedef ReceptorMemberCommand<Self>									EventPassThroughType;
		typedef typename EventPassThroughType::Pointer			EventPassThroughPointer;
				
		/** Check if any scales are set, and set the UseScales flag on or off; 
		 * after that call the superclass' implementation */
		virtual void StartOptimization(void);

		/** Methods to set parameters and print output at different stages
		 * in the registration process.*/
		virtual void BeforeRegistration(void);
		virtual void BeforeEachResolution(void);
		virtual void AfterEachResolution(void);
		virtual void AfterEachIteration(void);
		virtual void AfterRegistration(void);

		itkGetConstMacro(StartLineSearch, bool);
			
   			
	protected:

		ConjugateGradient();
		virtual ~ConjugateGradient() {};

		LineOptimizerPointer					m_LineOptimizer;

		/** Convert the line search stop condition to a string */
		virtual std::string GetLineSearchStopCondition(void) const;

		/** Generate a string, representing the phase of optimisation 
		 * (line search, main) */
		virtual std::string DeterminePhase(void) const;

	private:

		ConjugateGradient( const Self& );	// purposely not implemented
		void operator=( const Self& );							// purposely not implemented

		void InvokeIterationEvent(const EventObject & event);

		EventPassThroughPointer			m_EventPasser;
		double											m_SearchDirectionMagnitude;
		bool												m_StartLineSearch;
		bool												m_GenerateLineSearchIterations;
		bool												m_StopIfWolfeNotSatisfied;
			
	}; // end class ConjugateGradient
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxConjugateGradient.hxx"
#endif

#endif // end #ifndef __elxConjugateGradient_h

