#ifndef __elxAdvancedMattesMutualInformationMetric_HXX__
#define __elxAdvancedMattesMutualInformationMetric_HXX__

#include "elxAdvancedMattesMutualInformationMetric.h"
#include <string>

namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void AdvancedMattesMutualInformationMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of AdvancedMattesMutualInformation metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void AdvancedMattesMutualInformationMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** \todo Adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * 	metric->SetAlpha( config.GetAlpha(level) );
		 */

		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Get and set the number of histogram bins. */
		unsigned int numberOfHistogramBins = 32;
    this->GetConfiguration()->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", 0, true);
		this->GetConfiguration()->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", level );
		this->SetNumberOfHistogramBins( numberOfHistogramBins );

    /** Set whether a differentiable overlap should be used */
    std::string useDifferentiableOverlap = "true";
    this->GetConfiguration()->ReadParameter( useDifferentiableOverlap, "UseDifferentiableOverlap", 0, true );
    this->GetConfiguration()->ReadParameter( useDifferentiableOverlap, "UseDifferentiableOverlap", level );
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
      movingMaskInterpolationOrder, "MovingMaskInterpolationOrder", 0, true );
		this->GetConfiguration()->ReadParameter( 
      movingMaskInterpolationOrder, "MovingMaskInterpolationOrder", level );
		this->SetMovingImageMaskInterpolationOrder( 
      movingMaskInterpolationOrder );

    /** Get and set whether the metric should check if enough samples map inside the moving image. */
    std::string checkNumberOfSamples = "true";
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples, "CheckNumberOfSamples", 0, true );
    this->GetConfiguration()->ReadParameter( checkNumberOfSamples, "CheckNumberOfSamples", level );
    if ( checkNumberOfSamples == "false" )
    {
      this->SetCheckNumberOfSamples(false);
    }
    else
    {
      this->SetCheckNumberOfSamples(true);
    }

	} // end BeforeEachResolution
	
  
} // end namespace elastix


#endif // end #ifndef __elxAdvancedMattesMutualInformationMetric_HXX__
