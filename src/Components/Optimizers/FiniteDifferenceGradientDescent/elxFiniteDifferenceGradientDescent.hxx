#ifndef __elxFiniteDifferenceGradientDescent_hxx
#define __elxFiniteDifferenceGradientDescent_hxx

#include "elxFiniteDifferenceGradientDescent.h"
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
		FiniteDifferenceGradientDescent<TElastix>
		::FiniteDifferenceGradientDescent() 
	{
		this->m_ShowMetricValues = false;
	} // end Constructor
	

	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void FiniteDifferenceGradientDescent<TElastix>::
		BeforeRegistration(void)
	{
		std::string showMetricValues("false");
		this->GetConfiguration()->ReadParameter(
			showMetricValues, "ShowMetricValues", 0);
		if (showMetricValues == "false")
		{
			this->m_ShowMetricValues = false;
			this->SetComputeCurrentValue(this->m_ShowMetricValues);
		}
		else
		{
			this->m_ShowMetricValues = true;
			this->SetComputeCurrentValue(this->m_ShowMetricValues);
		}

		/** Add some target cells to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:Gain a_k");
		xout["iteration"].AddTargetCell("4:||Gradient||");

		/** Format them as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:Gain a_k"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;
		
	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void FiniteDifferenceGradientDescent<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
		unsigned int level = static_cast<unsigned int>(
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
		this->m_Configuration->ReadParameter( maximumNumberOfIterations , "MaximumNumberOfIterations", level );
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
		void FiniteDifferenceGradientDescent<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information */
		
		if (this->m_ShowMetricValues)
		{
			xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
		}
		else
		{
			xl::xout["iteration"]["2:Metric"]		<< "---";
		}
		xl::xout["iteration"]["3:Gain a_k"] << this->GetLearningRate();
		xl::xout["iteration"]["4:||Gradient||"] << this->GetGradientMagnitude();

		/** Select new spatial samples for the computation of the metric
		 * \todo You may also choose to select new samples after evaluation
		 * of the metric value */
		if ( this->GetNewSamplesEveryIteration() )
		{
			this->SelectNewSamples();
		}
		

	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void FiniteDifferenceGradientDescent<TElastix>
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
		void FiniteDifferenceGradientDescent<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		double bestValue;
		if (this->m_ShowMetricValues)
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
	 * ******************* StartOptimization ***********************
	 */

  template <class TElastix>
    void FiniteDifferenceGradientDescent<TElastix>
    ::StartOptimization(void)
	{

		/** Check if the entered scales are correct and != [ 1 1 1 ...] */

		this->SetUseScales(false);
		const ScalesType & scales = this->GetScales();
		if ( scales.GetSize() == this->GetInitialPosition().GetSize() )
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
	

} // end namespace elastix

#endif // end #ifndef __elxFiniteDifferenceGradientDescent_hxx

