#ifndef __elxStochasticQuasiNewton_hxx
#define __elxStochasticQuasiNewton_hxx

#include "elxStochasticQuasiNewton.h"
#include <iomanip>
#include <string>
#include "vnl/vnl_math.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		StochasticQuasiNewton<TElastix>
		::StochasticQuasiNewton() 
	{

		this->m_SearchDirectionMagnitude = 0.0;
		this->m_SinusScalesAmplitude = 0.0;
		this->m_SinusScalesFrequency = 1.0;

	} // end Constructor
	

	/**
	 * ***************** StartOptimization ************************
	 */
	
	template <class TElastix>
		void StochasticQuasiNewton<TElastix>::
		StartOptimization(void)
	{

		unsigned long numberOfParameters = this->GetInitialPosition().GetSize();
		double small_number1 = 1e-5;
		
		if (this->m_SinusScalesAmplitude > small_number1)
		{
			this->SetSinusScales(this->m_SinusScalesAmplitude, 
				this->m_SinusScalesFrequency, numberOfParameters);
		}
		
		
		/** Check if the entered scales are correct and != [ 1 1 1 ...] */

		this->SetUseScales(false);
		const ScalesType & scales = this->GetScales();
		if ( scales.GetSize() == numberOfParameters )
		{
      ScalesType unit_scales( scales.GetSize() );
			unit_scales.Fill(1.0);
			if (scales != unit_scales)
			{
				/** only then: */
				this->SetUseScales(true);
			}

		}

		this->Superclass1::StartOptimization();

	} //end StartOptimization


	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void StochasticQuasiNewton<TElastix>::
		BeforeRegistration(void)
	{

		using namespace xl;
		
		/** Add target cells to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:StepLength");
		xout["iteration"].AddTargetCell("4a:||Gradient||");
		xout["iteration"].AddTargetCell("4b:||SearchDir||");
		xout["iteration"].AddTargetCell("5:UpdateFactor");
	
		/** Format the metric and stepsize as floats */			
		xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xout["iteration"]["3:StepLength"] << std::showpoint << std::fixed;
		xout["iteration"]["4a:||Gradient||"] << std::showpoint << std::fixed;
		xout["iteration"]["4b:||SearchDir||"] << std::showpoint << std::fixed;
		xout["iteration"]["5:UpdateFactor"] << std::showpoint << std::fixed;

				
	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void StochasticQuasiNewton<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
		unsigned int level = static_cast<unsigned int>(
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
		this->m_Configuration->ReadParameter( maximumNumberOfIterations , "MaximumNumberOfIterations", level );
		this->SetMaximumNumberOfIterations( maximumNumberOfIterations );
		
		/** Set the length of the initial step, used if no Hessian estimate
		 * is available yet: searchdir = - steplength * g / |g|
		 * |g| optional, see NormalizeInitialGradients option
		 */ 
		double stepLength = 1.0; 
		this->m_Configuration->ReadParameter( stepLength,
			"StepLength", level );
		this->SetInitialStepLengthEstimate(stepLength);
    

		/** Set the number of initialization steps
		 * (number of iterations to use for estimating the initial hessian)
		 * If 0, the initial hessian will be set to the initial step length estimate.
		 */
		unsigned int numberOfInitializationSteps = 5;
		this->m_Configuration->ReadParameter( numberOfInitializationSteps,
			"NumberOfInitializationSteps", level );
		this->SetNumberOfInitializationSteps(numberOfInitializationSteps);

		this->m_SearchDirectionMagnitude = 0.0;

		double betaMax = 2.0;
		double decay_A = 50;
		double decay_alpha = 0.602;
		this->m_Configuration->ReadParameter( betaMax, "BetaMax", level );
		double detMax = betaMax;
		this->m_Configuration->ReadParameter( detMax, "DetMax", level );
		this->m_Configuration->ReadParameter( decay_A, "Decay_A", level );
		this->m_Configuration->ReadParameter( decay_alpha, "Decay_alpha", level );
		this->SetBetaMax(betaMax);
		this->SetDetMax(detMax);
		this->SetDecay_A(decay_A);
		this->SetDecay_alpha(decay_alpha);

		/** Setting: whether to normalize the initial gradients or not */
		bool normalizeInitialGradientsBool = true;
		std::string normalizeInitialGradients = "true";
		this->GetConfiguration()->
			ReadParameter(normalizeInitialGradients, "NormalizeInitialGradients", level);
		if (normalizeInitialGradients == "true")
		{
			normalizeInitialGradientsBool = true;
		}
		else
		{
			normalizeInitialGradientsBool = false;
		}
		this->SetNormalizeInitialGradients(normalizeInitialGradientsBool);

		/** Setting: the amplitude and frequency of a sinus that controls the scales */
		this->m_SinusScalesAmplitude = 0.0;
		this->m_SinusScalesFrequency = 1.0;
		this->GetConfiguration()->
			ReadParameter(this->m_SinusScalesAmplitude, "SinusScalesAmplitude", level);
		this->GetConfiguration()->
			ReadParameter(this->m_SinusScalesFrequency, "SinusScalesFrequency", level);

		/** Set the NumberOfGradientDescentIterations.*/
		unsigned int numberOfGradientDescentIterations = 50;
		this->m_Configuration->ReadParameter( numberOfGradientDescentIterations , "NumberOfGradientDescentIterations", level );
		this->SetNumberOfGradientDescentIterations( numberOfGradientDescentIterations );
		

					
	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void StochasticQuasiNewton<TElastix>
		::AfterEachIteration(void)
	{
		
		using namespace xl;

		/** Print some information. */
		
		xout["iteration"]["2:Metric"]	<<
			this->GetCurrentValue();
		xout["iteration"]["3:StepLength"] << 
			this->GetCurrentStepLength(); 
		xout["iteration"]["4a:||Gradient||"] << 
	    this->GetCurrentGradient().magnitude();
		xout["iteration"]["4b:||SearchDir||"] << 
			this->m_SearchDirectionMagnitude ;
		xout["iteration"]["5:UpdateFactor"] << 
			this->GetUpdateFactor();
	
		
		/** If new samples: */
		if ( this->GetNewSamplesEveryIteration() )
		{
			this->SelectNewSamples();
		}
		
	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void StochasticQuasiNewton<TElastix>
		::AfterEachResolution(void)
	{
		/**
    typedef enum {
      MetricError,
      MaximumNumberOfIterations,
      GradientMagnitudeTolerance,
			ZeroStep,
      Unknown } 
			*/
		
		std::string stopcondition;
		
		switch( this->GetStopCondition() )
		{
	
			case MetricError :
			  stopcondition = "Error in metric";	
			  break;	
	  
			case MaximumNumberOfIterations :
			  stopcondition = "Maximum number of iterations has been reached";	
			  break;	
	
			case GradientMagnitudeTolerance :
			  stopcondition = "The gradient magnitude has (nearly) vanished";	
			  break;	

			case ZeroStep :
				stopcondition = "The last step size was (nearly) zero";
				break;
					
		  default:
			  stopcondition = "Unknown";
			  break;
		}

		/** Print the stopping condition */
		elxout << "Stopping condition: " << stopcondition << "." << std::endl;

	} // end AfterEachResolution
	
	/**
	 * ******************* AfterRegistration ************************
	 */

	template <class TElastix>
		void StochasticQuasiNewton<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		
		double bestValue = this->GetCurrentValue();
		elxout
			<< std::endl
			<< "Final metric value  = " 
			<< bestValue
			<< std::endl;
		
	} // end AfterRegistration


	/**
	 * ******************* ComputeSearchDirection **********************
	 */

	template <class TElastix>
	void StochasticQuasiNewton<TElastix>
	::ComputeSearchDirection(
    const DerivativeType & gradient,
    ParametersType & searchDir)
	{
		this->Superclass1::ComputeSearchDirection(gradient,searchDir);
		this->m_SearchDirectionMagnitude = searchDir.magnitude();
	} // end ComputeSearchDirection


} // end namespace elastix

#endif // end #ifndef __elxStochasticQuasiNewton_hxx



