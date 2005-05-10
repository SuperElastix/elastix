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
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int NormalizedCorrelationMetric<TElastix>
		::BeforeAll(void)
	{
		/** Return a value.*/
		return 0;

	} // end BeforeAll


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


	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>::
		BeforeRegistration(void)
	{	
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void NormalizedCorrelationMetric<TElastix>::
		BeforeEachResolution(void)
	{
	} // end BeforeEachResolution
	
	
} // end namespace elastix


#endif // end #ifndef __elxNormalizedCorrelationMetric_HXX__

