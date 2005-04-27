#ifndef __elxStandardGradientDescent_hxx
#define __elxStandardGradientDescent_hxx

#include "elxStandardGradientDescent.h"
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
		StandardGradientDescent<TElastix>
		::StandardGradientDescent() 
	{
		
	} // end Constructor
	

	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void StandardGradientDescent<TElastix>::
		BeforeRegistration(void)
	{
		
		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:StepSize");
		xout["iteration"].AddTargetCell("4:||Gradient||");

		/** Format the metric and stepsize as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:StepSize"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;

	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void StandardGradientDescent<TElastix>
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
		double A;
		double alpha;
		
		if (level == 0)
		{
			a = 400;
			A = 50.0;
			alpha = 0.602;
		}
		else 
		{
			/** If only one parameter is set, then this parameter
			 * is used in each resolution.
			 */
			a = this->GetParam_a();
			A = this->GetParam_A();
			alpha = this->GetParam_alpha();
		}

		this->GetConfiguration()->ReadParameter(a, "SP_a", level);
		this->GetConfiguration()->ReadParameter(A, "SP_A", level);
		this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", level);
		
		this->SetParam_a(	a );
		this->SetParam_A( A );
		this->SetParam_alpha( alpha );
				
	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void StandardGradientDescent<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information */
		xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
		xl::xout["iteration"]["3:StepSize"] << this->GetLearningRate();
		xl::xout["iteration"]["4:||Gradient||"] << this->GetGradientMagnitude();

		/** Select new spatial samples for the computation of the metric */
		if ( this->GetNewSamplesEveryIteration() )
		{
			this->SelectNewSamples();
		}

	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void StandardGradientDescent<TElastix>
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
		void StandardGradientDescent<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		
		double bestValue = this->GetValue();
		elxout
			<< std::endl
			<< "Final metric value  = " 
			<< bestValue
			<< std::endl;
		
	} // end AfterRegistration


	/**
	 * ******************* SetInitialPosition ***********************
	 */

	template <class TElastix>
		void StandardGradientDescent<TElastix>
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

#endif // end #ifndef __elxStandardGradientDescent_hxx

