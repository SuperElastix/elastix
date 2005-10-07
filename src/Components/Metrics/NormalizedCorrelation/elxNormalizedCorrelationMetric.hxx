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

