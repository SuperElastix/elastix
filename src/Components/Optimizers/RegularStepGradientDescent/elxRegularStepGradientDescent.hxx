#ifndef __elxRegularStepGradientDescent_hxx
#define __elxRegularStepGradientDescent_hxx

#include "elxRegularStepGradientDescent.h"
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
		RegularStepGradientDescent<TElastix>
		::RegularStepGradientDescent() 
	{
	} // end Constructor
	

	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void RegularStepGradientDescent<TElastix>::
		BeforeRegistration(void)
	{
		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("1:ItNr");
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
		void RegularStepGradientDescent<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level.*/
		unsigned int level = static_cast<unsigned int>(
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
		
		/** Set the Gradient Magnitude Stopping Criterion.*/
		double minGradientMagnitude = 1e-8;
		this->m_Configuration->ReadParameter( minGradientMagnitude, "MinimumGradientMagnitude", level );
		this->SetGradientMagnitudeTolerance( minGradientMagnitude );

		/** Set the MaximumStepLength.*/
		double maxStepLength = 16.0 / pow( 2.0, static_cast<int>( level ) );
		this->m_Configuration->ReadParameter( maxStepLength, "MaximumStepLength", level );
		this->SetMaximumStepLength( maxStepLength );

		/** Set the MinimumStepLength.*/
		double minStepLength = 0.5 / pow( 2.0, static_cast<int>( level ) );
		this->m_Configuration->ReadParameter( minStepLength, "MinimumStepLength", level );
		this->SetMinimumStepLength( minStepLength );

		/** \todo max and min steplength should maybe depend on the imagespacing or on something else... */
		
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
		this->m_Configuration->ReadParameter( maximumNumberOfIterations , "MaximumNumberOfIterations", level );
		this->SetNumberOfIterations( maximumNumberOfIterations );

		/** \todo put in SPoptimizer:
		if (SP)
		{
			optimizer->SetParam_a(
				config.GetParam_a(level));
			optimizer->SetParam_c(
				config.GetParam_c(level));
			optimizer->SetParam_A(
				config.GetParam_A(level));
			optimizer->SetParam_alpha(
				config.GetParam_alpha(level));
			optimizer->SetParam_gamma(
				config.GetParam_gamma(level));
		}
		*/

	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void RegularStepGradientDescent<TElastix>
		::AfterEachIteration(void)
	{
		/** Print some information */
		xl::xout["iteration"]["1:ItNr"]			<< this->GetCurrentIteration();
		xl::xout["iteration"]["2:Metric"]		<< this->GetValue();
		xl::xout["iteration"]["3:StepSize"] << this->GetCurrentStepLength();
		xl::xout["iteration"]["4:||Gradient||"] << this->GetGradientMagnitude();
	} // end AfterEachIteration


	/**
	 * ***************** AfterEachResolution *************************
	 */

	template <class TElastix>
		void RegularStepGradientDescent<TElastix>
		::AfterEachResolution(void)
	{
		
		/**
		 * enum   StopConditionType {   GradientMagnitudeTolerance = 1, StepTooSmall,
		 * ImageNotAvailable, SamplesNotAvailable, MaximumNumberOfIterations  
		 */
		std::string stopcondition;

		switch( this->GetStopCondition() )
		{
			
		case GradientMagnitudeTolerance :
			stopcondition = "Minimum gradient magnitude has been reached";
			break;
			
		case StepTooSmall :
			stopcondition = "Minimum step size has been reached";	
			break;

		case MaximumNumberOfIterations :
			stopcondition = "Maximum number of iterations has been reached";	
			break;	
		
		case ImageNotAvailable :
			stopcondition = "No image available";	
			break;	
		
		case SamplesNotAvailable :
			stopcondition = "No samples available";	
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
		void RegularStepGradientDescent<TElastix>
		::AfterRegistration(void)
	{
	  /** Print the best metric value */
		double bestValue = this->GetValue();
		elxout << std::endl << "Final metric value  = " << bestValue  << std::endl;

	} // end AfterRegistration


	/**
	 * ******************* SetInitialPosition ***********************
	 */

	template <class TElastix>
		void RegularStepGradientDescent<TElastix>
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

#endif // end #ifndef __elxRegularStepGradientDescent_hxx

