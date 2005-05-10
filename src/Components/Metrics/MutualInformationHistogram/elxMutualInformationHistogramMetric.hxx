#ifndef __elxMutualInformationHistogramMetric_HXX__
#define __elxMutualInformationHistogramMetric_HXX__

#include "elxMutualInformationHistogramMetric.h"

namespace elastix
{
using namespace itk;


	/**
	 * ********************* Constructor ****************************
	 */

	template <class TElastix>
		MutualInformationHistogramMetric<TElastix>
		::MutualInformationHistogramMetric()
	{
	} // end Constructor


	/**
	 * ************************ BeforeAll ***************************
	 */
	
	template <class TElastix>
		int MutualInformationHistogramMetric<TElastix>
		::BeforeAll(void)
	{
		/** Return a value.*/
		return 0;

	} // end BeforeAll


	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MutualInformationHistogramMetric metric took: "
			<< static_cast<long>(timer->GetElapsedClockSec() *1000) << " ms." << std::endl;

	} // end Initialize

	
	/**
	 * ******************* BeforeRegistration ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		BeforeRegistration(void)
	{		
	} // end BeforeRegistration
	

	/**
	 * ***************** BeforeEachResolution ***********************
	 */

	template <class TElastix>
		void MutualInformationHistogramMetric<TElastix>::
		BeforeEachResolution(void)
	{
		/** \todo adapt SecondOrderRegularisationMetric.
		 * Set alpha, which balances the similarity and deformation energy
		 * E_total = (1-alpha)*E_sim + alpha*E_def.
		 * metric->SetAlpha( config.GetAlpha(level) );
		 */

		unsigned int nrOfParameters = this->m_Elastix->GetElxTransformBase()
			->GetAsITKBaseType()->GetNumberOfParameters();
		ScalesType derivativeStepLengthScales( nrOfParameters );
		derivativeStepLengthScales.Fill( 1.0 );

		/** Read the parameters from the ParameterFile.*
		this->m_Configuration->ReadParameter( histogramSize, "HistogramSize", 0 );
		this->m_Configuration->ReadParameter( paddingValue, "PaddingValue", 0 );
		this->m_Configuration->ReadParameter( derivativeStepLength, "DerivativeStepLength", 0 );
		this->m_Configuration->ReadParameter( derivativeStepLengthScales, "DerivativeStepLengthScales", 0 );
		this->m_Configuration->ReadParameter( upperBoundIncreaseFactor, "UpperBoundIncreaseFactor", 0 );
		this->m_Configuration->ReadParameter( usePaddingValue, "UsePaddingValue", 0 );
		*/
		/** Set them.*/
		//this->SetHistogramSize( ?? );
		//this->SetPaddingValue( ?? );
		//this->SetDerivativeStepLength( ?? );
		this->SetDerivativeStepLengthScales( derivativeStepLengthScales );
		//this->SetUpperBoundIncreaseFactor( ?? );
		//this->SetUsePaddingValue( ?? );
			
	} // end BeforeEachResolution
	
	
} // end namespace elastix


#endif // end #ifndef __elxMutualInformationHistogramMetric_HXX__

