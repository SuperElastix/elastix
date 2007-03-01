#ifndef __elxMattesMutualInformationMetric_HXX__
#define __elxMattesMutualInformationMetric_HXX__

#include "elxMattesMutualInformationMetric.h"
#include <string>

namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MattesMutualInformation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MattesMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Get and set the number of histogram bins. */
		unsigned int numberOfHistogramBins = 32;
    this->GetConfiguration()->ReadParameter( numberOfHistogramBins,
      "NumberOfHistogramBins", this->GetComponentLabel(), level, 0 );
		this->SetNumberOfHistogramBins( numberOfHistogramBins );

    /** Get and set whether the metric should check if enough samples map inside the moving image. */
    bool checkNumberOfSamples = true;
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples, 
      "CheckNumberOfSamples", this->GetComponentLabel(), level, 0 );
    this->SetCheckNumberOfSamples( checkNumberOfSamples );

	} // end BeforeEachResolution
	
  
} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationMetric_HXX__

