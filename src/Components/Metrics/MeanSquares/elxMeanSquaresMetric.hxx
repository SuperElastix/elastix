#ifndef __elxMeanSquaresMetric_HXX__
#define __elxMeanSquaresMetric_HXX__

#include "elxMeanSquaresMetric.h"


namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MeanSquaresMetric<TElastix>::MeanSquaresMetric()
	{
	} // end Constructor


	/**
	 * ************************ BeforeRegistration ***************************
	 */
	
	template <class TElastix>
		void MeanSquaresMetric<TElastix>
		::BeforeRegistration(void)
	{
		/** Get and set UseAllPixels. Default true. */
		std::string useAllPixels = "true";
		this->GetConfiguration()->ReadParameter( useAllPixels, "UseAllPixels", 0 );
		if ( useAllPixels == "false" ) this->SetUseAllPixels( false );
		else this->SetUseAllPixels( true );

	} // end BeforeRegistration


	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MeanSquaresMetric<TElastix>
		::BeforeEachResolution(void)
	{
		/** Get the current resolution level. */
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();

		/** Get and set NumberOfSpatialSamples. This only makes sense
		 * if UseAllPixels is true. */
		unsigned long numberOfSpatialSamples = 5000;
		this->GetConfiguration()->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", 0 );
		this->GetConfiguration()->ReadParameter( numberOfSpatialSamples, "NumberOfSpatialSamples", level );
		this->SetNumberOfSpatialSamples( numberOfSpatialSamples );

	} // end BeforeEachResolution
	

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MeanSquaresMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MeanSquares metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


} // end namespace elastix


#endif // end #ifndef __elxMeanSquaresMetric_HXX__

