#ifndef __elxNormalizedCorrelationMetric_HXX__
#define __elxNormalizedCorrelationMetric_HXX__

#include "elxNormalizedCorrelationMetric.h"


namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		NormalizedCorrelationMetric<TElastix>::NormalizedCorrelationMetric()
	{
	} // end Constructor


	/**
	 * ************************ BeforeRegistration ***************************
	 */
	
	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>
		::BeforeRegistration(void)
	{
	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

		/** Get and set SubtractMean. Default true. */
		std::string subtractMean = "true";
		this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", 0 );
		this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean", level );
		if ( subtractMean == "false" ) this->SetSubtractMean( false );
		else this->SetSubtractMean( true );
		
	} // end BeforeEachResolution
	

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of NormalizedCorrelation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


} // end namespace elastix


#endif // end #ifndef __elxNormalizedCorrelationMetric_HXX__

