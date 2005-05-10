#ifndef __elxViolaWellsMutualInformationMetric_HXX__
#define __elxViolaWellsMutualInformationMetric_HXX__

#include "elxViolaWellsMutualInformationMetric.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		ViolaWellsMutualInformationMetric<TElastix>
		::ViolaWellsMutualInformationMetric()
	{
	} // end Constructor


	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int ViolaWellsMutualInformationMetric<TElastix>
		::BeforeAll(void)
	{
		/** Return a value.*/
		return 0;

	} // end BeforeAll


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void ViolaWellsMutualInformationMetric<TElastix>::
		Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of ViolaWellsMutualInformationMetric metric took: "
			<< static_cast<long>(timer->GetElapsedClockSec() *1000) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void ViolaWellsMutualInformationMetric<TElastix>::
		BeforeRegistration(void)
	{		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void ViolaWellsMutualInformationMetric<TElastix>::
		BeforeEachResolution(void)
	{
		/** \todo adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * metric->SetAlpha( config.GetAlpha(level) );
		 */

		/** Get the current resolution level.*/
		unsigned int level = 
			( this->m_Registration->GetAsITKBaseType() )->GetCurrentLevel();
		
		/** Set the number of histogram bins and spatial samples.*/
		unsigned int numberOfSpatialSamples = 10000;
		/** \todo guess the default numberOfSpatialSamples from the 
		 * imagesize, the numberOfParameters, and the number of bins....
		 */

		/** Set the intensity standard deviation of the fixed
		 * and moving images. This defines the kernel bandwidth
		 * used in the joint probability distribution calculation.
		 * Default value is 0.4 which works well for image intensities
		 * normalized to a mean of 0 and standard deviation of 1.0.
		 * Value is clamped to be always greater than zero.
		 */
		double fixedImageStandardDeviation = 0.4;
		double movingImageStandardDeviation = 0.4;
		/** \todo calculate them??? */
		
		/** Read the parameters from the ParameterFile.*/
		this->m_Configuration->ReadParameter(
			numberOfSpatialSamples, "NumberOfSpatialSamples", level );
		this->m_Configuration->ReadParameter(
			fixedImageStandardDeviation, "FixedImageStandardDeviation", level );
		this->m_Configuration->ReadParameter(
			movingImageStandardDeviation, "MovingImageStandardDeviation", level );
		
		/** Set them.*/
		this->SetNumberOfSpatialSamples( numberOfSpatialSamples );
		this->SetFixedImageStandardDeviation( fixedImageStandardDeviation );
		this->SetMovingImageStandardDeviation( movingImageStandardDeviation );
					
	} // end BeforeEachResolution
	
	
} // end namespace elastix


#endif // end #ifndef __elxViolaWellsMutualInformationMetric_HXX__

