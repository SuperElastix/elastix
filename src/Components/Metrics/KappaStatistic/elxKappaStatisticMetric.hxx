#ifndef __elxKappaStatisticMetric_HXX__
#define __elxKappaStatisticMetric_HXX__

#include "elxKappaStatisticMetric.h"


namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void KappaStatisticMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of KappaStatistic metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


  /**
	 * ***************** BeforeRegistration ***********************
	 */

	template <class TElastix>
		void KappaStatisticMetric<TElastix>
		::BeforeRegistration(void)
	{
    /** Get and set taking the complement. */
    bool useComplement = false;
    this->GetConfiguration()->ReadParameter( useComplement,
      "UseComplement", this->GetComponentLabel(), 0, -1 );
    this->SetComplement( useComplement );

    /** Get and set the foreground value. */
    double foreground = 1.0;
    this->GetConfiguration()->ReadParameter( foreground,
      "ForegroundValue", this->GetComponentLabel(), 0, -1 );
    this->SetForegroundValue( foreground );

  } // end BeforeRegistration()


} // end namespace elastix


#endif // end #ifndef __elxKappaStatisticMetric_HXX__

