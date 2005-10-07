#ifndef __elxSimultaneousPerturbation_hxx
#define __elxSimultaneousPerturbation_hxx

#include "elxSimultaneousPerturbation.h"
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
		SimultaneousPerturbation<TElastix>
		::SimultaneousPerturbation() 
	{
		this->m_ShowMetricValues = false;
	} // end Constructor
	

	/**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>::
		BeforeRegistration(void)
	{
		std::string showMetricValues("false");
		this->GetConfiguration()->ReadParameter(
			showMetricValues, "ShowMetricValues", 0);
		if (showMetricValues == "false")
		{
			this->m_ShowMetricValues = false;
		}
		else
		{
			this->m_ShowMetricValues = true;
		}

		/** Add the target cell "stepsize" to xout["iteration"].*/
		xout["iteration"].AddTargetCell("2:Metric");
		xout["iteration"].AddTargetCell("3:Gain a_k");
		xout["iteration"].AddTargetCell("4:||Gradient||");

		/** Format the metric and stepsize as floats */			
		xl::xout["iteration"]["2:Metric"]		<< std::showpoint << std::fixed;
		xl::xout["iteration"]["3:Gain a_k"] << std::showpoint << std::fixed;
		xl::xout["iteration"]["4:||Gradient||"] << std::showpoint << std::fixed;
		
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
			this->m_Registration->GetAsITKBaseType()->GetCurrentLevel() );
				
		/** Set the maximumNumberOfIterations.*/
		unsigned int maximumNumberOfIterations = 100;
		this->m_Configuration->ReadParameter( maximumNumberOfIterations , "MaximumNumberOfIterations", level );
		this->SetMaximumNumberOfIterations( maximumNumberOfIterations );

		/** Set the number of perturbation used to construct a gradient estimate g_k. */
    unsigned int numberOfPerturbations = 1;
		this->m_Configuration->ReadParameter( numberOfPerturbations , "NumberOfPerturbations", level );
		this->SetNumberOfPerturbations( numberOfPerturbations );

		double a;
		double c;
		double A;
		double alpha;
		double gamma;

		/** \todo call the GuessParameters function */
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
			a = this->Geta();
			c = this->Getc();
			A = this->GetA();
			alpha = this->GetAlpha();
			gamma = this->GetGamma();
		}

		this->GetConfiguration()->ReadParameter(a, "SP_a", level);
		this->GetConfiguration()->ReadParameter(c, "SP_c", level);
		this->GetConfiguration()->ReadParameter(A, "SP_A", level);
		this->GetConfiguration()->ReadParameter(alpha, "SP_alpha", level);
		this->GetConfiguration()->ReadParameter(gamma, "SP_gamma", level);

		this->Seta(	a );
		this->Setc( c );
		this->SetA( A );
		this->SetAlpha( alpha );
		this->SetGamma( gamma );

		/** Ignore the build-in stop criterion; it's quite ad hoc. */
		this->SetTolerance(0.0);
		
	} // end BeforeEachResolution


	/**
	 * ***************** AfterEachIteration *************************
	 */

	template <class TElastix>
		void SimultaneousPerturbation<TElastix>
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
		void SimultaneousPerturbation<TElastix>
		::AfterEachResolution(void)
	{
		
		/**
		 * enum   StopConditionType {  MaximumNumberOfIterations, MetricError }  
		 * ignore the BelowTolerance-criterion.
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
		
		bestValue = this->GetValue();
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

