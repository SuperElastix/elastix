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

    /** Set whether a differentiable overlap should be used */
    std::string useDifferentiableOverlap = "true";
    this->GetConfiguration()->ReadParameter( useDifferentiableOverlap, "UseDifferentiableOverlap", 0 );
    this->GetConfiguration()->ReadParameter( useDifferentiableOverlap, "UseDifferentiableOverlap", level, true );
    if ( useDifferentiableOverlap == "false" )
    {
      this->SetUseDifferentiableOverlap(false);
    }
    else
    {
      this->SetUseDifferentiableOverlap(true);
    }

    /** Get and set the mask interpolation order */
		unsigned int movingMaskInterpolationOrder = 2;
    this->GetConfiguration()->ReadParameter( 
      movingMaskInterpolationOrder, "MovingMaskInterpolationOrder", 0 );
		this->GetConfiguration()->ReadParameter( 
      movingMaskInterpolationOrder, "MovingMaskInterpolationOrder", level, true );
		this->SetMovingImageMaskInterpolationOrder( 
      movingMaskInterpolationOrder );

    /** Get and set whether the metric should check if enough samples map inside the moving image. */
    std::string checkNumberOfSamples = "true";
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples, "CheckNumberOfSamples", 0, true );
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples, "CheckNumberOfSamples", level );
    if ( checkNumberOfSamples == "false" )
    {
      this->SetRequiredRatioOfValidSamples(0.0);
    }
    else
    {
      this->SetRequiredRatioOfValidSamples(0.25);
    }
		
  } // end BeforeEachResolution



} // end namespace elastix


#endif // end #ifndef __elxAdvancedMeanSquaresMetric_HXX__

