#ifndef __elxStochasticQuasiNewton_h
#define __elxStochasticQuasiNewton_h

#include "itkStochasticQuasiNewtonOptimizer.h"
#include "elxIncludes.h"

namespace elastix
{
using namespace itk;


	/**
	 * **************** StochasticQuasiNewton ******************
	 *
	 * The StochasticQuasiNewton class ....
	 */

	template <class TElastix>
		class StochasticQuasiNewton :
		public
			itk::StochasticQuasiNewtonOptimizer,
		public
			OptimizerBase<TElastix>
	{
	public:

		/** Standard ITK.*/
		typedef StochasticQuasiNewton						  	Self;
		typedef StochasticQuasiNewtonOptimizer			Superclass1;
		typedef OptimizerBase<TElastix>							Superclass2;
		typedef SmartPointer<Self>									Pointer;
		typedef SmartPointer<const Self>						ConstPointer;
		
		/** Method for creation through the object factory. */
		itkNewMacro( Self );
		
		/** Run-time type information (and related methods). */
		itkTypeMacro( StochasticQuasiNewton, StochasticQuasiNewtonOptimizer );
		
		/** Name of this class.*/
		elxClassNameMacro( "StochasticQuasiNewton" );

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

   			
	protected:

		StochasticQuasiNewton();
		virtual ~StochasticQuasiNewton() {};

		/** Steal the searchDir.magnitude() */
		virtual void ComputeSearchDirection(
      const DerivativeType & gradient,
      ParametersType & searchDir);

	private:

		StochasticQuasiNewton( const Self& );	// purposely not implemented
		void operator=( const Self& );							// purposely not implemented

		double											m_SearchDirectionMagnitude;
			
	}; // end class StochasticQuasiNewton
	

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxStochasticQuasiNewton.hxx"
#endif

#endif // end #ifndef __elxStochasticQuasiNewton_h

