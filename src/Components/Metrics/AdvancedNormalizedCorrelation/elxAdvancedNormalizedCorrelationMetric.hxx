#ifndef __elxAdvancedNormalizedCorrelationMetric_HXX__
#define __elxAdvancedNormalizedCorrelationMetric_HXX__

#include "elxAdvancedNormalizedCorrelationMetric.h"


namespace elastix
{
using namespace itk;


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void AdvancedNormalizedCorrelationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Get and set whether the metric should check if enough samples map inside the moving image. */
    bool checkNumberOfSamples = true;
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples,
      "CheckNumberOfSamples", this->GetComponentLabel(), level, 0 );
    if ( !checkNumberOfSamples )
    {
      this->SetRequiredRatioOfValidSamples(0.0);
    }
    else
    {
      this->SetRequiredRatioOfValidSamples(0.25);
    }

		/** Get and set SubtractMean. Default true. */
		bool subtractMean = true;
		this->GetConfiguration()->ReadParameter( subtractMean, "SubtractMean",
      this->GetComponentLabel(), level, 0 );
		this->SetSubtractMean( subtractMean );
		
	} // end BeforeEachResolution
	

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void AdvancedNormalizedCorrelationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of AdvancedNormalizedCorrelation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


} // end namespace elastix


#endif // end #ifndef __elxAdvancedNormalizedCorrelationMetric_HXX__

