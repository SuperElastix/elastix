#ifndef __elxSimultaneousPerturbation_hxx
#define __elxSimultaneousPerturbation_hxx

#include "elxSimultaneousPerturbation.h"
#include <iomanip>
#include <string>
#include "math.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		SimultaneousPerturbation<TElastix>
		::SimultaneousPerturbation() 
	{
		m_ShowMetricValues = true;
	} // end Constructor
	

	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>::
		BeforeRegistration(void)
	{
		std::string showMetricValues("true");
		this->GetConfiguration()->ReadParameter(
			showMetricValues, "ShowMetricValues", 0);
		if (showMetricValues == "false")
		{
			m_ShowMetricValues = false;
			this->SetComputeCurrentValue(m_ShowMetricValues);
		}
		else
		{
			m_ShowMetricValues = true;
			this->SetComputeCurrentValue(m_ShowMetricValues);
		}

		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("1:ItNr");
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:StepSize");
		xout["iteration"].AddTargetCell("4:Gain a_k");
		xout["iteration"].AddTargetCell("5:||Gradient||");

		/** Format the metric and stepsize as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:StepSize"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:Gain a_k"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["5:||Gradient||"] << std::showpoint << std::fixed;
		
	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
		unsigned int level = static_cast<unsigned int>(
			m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
		m_Configuration->ReadParameter( maximumNumberOfIterations , "MaximumNumberOfIterations", level );
		this->SetNumberOfIterations( maximumNumberOfIterations );

		double a;
		double c;
		double A;
		double alpha;
		double gamma;

		if (level == 0)
		{
			a = 400;
			c = 1.0;
			A = 50.0;
			alpha = 0.602;
			gamma = 0.101;
		}
		else 
		{
			/** If only one parameter is set, then this parameter
			 * is used in each resolution.
			 */
			a = this->GetParam_a();
			c = this->GetParam_c();
			A = this->GetParam_A();
			alpha = this->GetParam_alpha();
			gamma = this->GetParam_gamma();
		}

		this->GetConfiguration()->ReadParameter(a, "SP_a", level);
		this->GetConfiguration()->ReadParameter(c, "SP_c", level);
		this->GetConfiguration()->ReadParameter(A, "SP_A", level);
		this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", level);
		this->GetConfiguration()->ReadParameter(gamma, "SP_gamma", level);

		this->SetParam_a(	a );
		this->SetParam_c( c );
		this->SetParam_A( A );
		this->SetParam_alpha( alpha );
		this->SetParam_gamma( gamma );
		
	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information */
		xl::xout["iteration"]["1:ItNr"]			<< this->GetCurrentIteration();

		if (m_ShowMetricValues)
		{
			xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
		}
		else
		{
			xl::xout["iteration"]["2:Metric"]		<< "---";
		}
		xl::xout["iteration"]["3:StepSize"] << this->GetCurrentStepLength();
		xl::xout["iteration"]["4:Gain a_k"] << this->GetLearningRate();
		xl::xout["iteration"]["5:||Gradient||"] << this->GetGradientMagnitude();
		

	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>
		::AfterEachResolution(void)
	{
		
		/**
		 * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }  
		 */
		std::string stopcondition;

		
		switch( this->GetStopCondition() )
		{
	
		case MaximumNumberOfIterations :
			stopcondition = "Maximum number of iterations has been reached";	
			break;	
		
		case MetricError :
			stopcondition = "Error in metric";	
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
		void SimultaneousPerturbation<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		double bestValue;
		if (m_ShowMetricValues)
		{
			bestValue = this->GetValue();
			elxout
				<< std::endl
				<< "Final metric value  = " 
				<< bestValue
				<< std::endl;
		}
		else
		{
			elxout 
			<< std::endl
			<< "Run Elastix again with the option \"ShowMetricValues\" set"
			<< " to \"true\", to see information about the metric values. "
			<< std::endl;
		}

	} // end AfterRegistration


	/**
	 * ******************* SetInitialPosition ***********************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>
		::SetInitialPosition( const ParametersType & param )
	{
		/** Override the implementation in itkOptimizer.h, to
		 * ensure that the scales array and the parameters
		 * array have the same size.
		 */

		/** Call the Superclass' implementation. */
		this->Superclass1::SetInitialPosition( param );

		/** Set the scales array to the same size if the size has been changed */
		ScalesType scales = this->GetScales();
		unsigned int paramsize = param.Size();

		if ( ( scales.Size() ) != paramsize )
		{
			ScalesType newscales( paramsize );
			newscales.Fill(1.0);
			this->SetScales( newscales );
		}
		
		/** \todo to optimizerbase? */

	} // end SetInitialPosition
	

} // end namespace elastix

#endif // end #ifndef __elxSimultaneousPerturbation_hxx

