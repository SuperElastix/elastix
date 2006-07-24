#ifndef __elxMattesMutualInformationMetric_HXX__
#define __elxMattesMutualInformationMetric_HXX__

#include "elxMattesMutualInformationMetric.h"
#include "vnl/vnl_math.h"
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
    this->m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", 0 );
		this->m_Configuration->ReadParameter( numberOfHistogramBins, "NumberOfHistogramBins", level, true );
		
		this->SetNumberOfHistogramBins( numberOfHistogramBins );

	} // end BeforeEachResolution
	
  
} // end namespace elastix


#endif // end #ifndef __elxMattesMutualInformationMetric_HXX__

