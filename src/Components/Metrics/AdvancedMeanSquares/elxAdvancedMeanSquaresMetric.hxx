#ifndef __elxAdvancedMeanSquaresMetric_HXX__
#define __elxAdvancedMeanSquaresMetric_HXX__

#include "elxAdvancedMeanSquaresMetric.h"


namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void AdvancedMeanSquaresMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of AdvancedMeanSquares metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


  /**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void AdvancedMeanSquaresMetric<TElastix>
		::BeforeEachResolution(void)
	{
    /** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

    /** Get and set whether the metric should check if enough samples map inside the moving image. */
    std::string checkNumberOfSamples = "true";
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples,
      "CheckNumberOfSamples", this->GetComponentLabel(), level, 0 );
    if ( checkNumberOfSamples == "false" )
    {
      this->SetRequiredRatioOfValidSamples(0.0);
    }
    else
    {
      this->SetRequiredRatioOfValidSamples(0.25);
    }

    /** Get and set the normalization. */
    bool useNormalization = false;
    this->GetConfiguration()->ReadParameter( useNormalization,
      "UseNormalization", this->GetComponentLabel(), level, 0 );
    this->SetUseNormalization( useNormalization );
		
  } // end BeforeEachResolution


} // end namespace elastix


#endif // end #ifndef __elxAdvancedMeanSquaresMetric_HXX__

