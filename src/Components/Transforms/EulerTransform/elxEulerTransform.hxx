#ifndef __elxEulerTransform_HXX_
#define __elxEulerTransform_HXX_

#include "elxEulerTransform.h"

namespace elastix
{
	using namespace itk;
	
	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		EulerTransformElastix<TElastix>
		::EulerTransformElastix()
	{
	} // end Constructor
	
	
	/**
	 * ******************* BeforeRegistration ***********************
	 */
	
	template <class TElastix>
		void EulerTransformElastix<TElastix>
		::BeforeRegistration(void)
	{
		/** Task 1 - Set initial parameters.*/
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill(0.0);

		/** And give it to m_Registration.*/
		m_Registration->GetAsITKBaseType()
			->SetInitialTransformParameters( dummyInitialParameters );
		
		/** Task 2 - Set the scales.*/
		ScalesType newscales( this->GetNumberOfParameters() );
		newscales.Fill(1.0);
		double scaler = 100000.0;
		m_Configuration->ReadParameter( scaler, "Scaler", 0 );
		/**
		 * - If the Dimension is 3, the first 3 parameters represent angles.
		 * - If the Dimension is 2, only the first parameter represent an angle.
		 */
		unsigned int AnglePartEnd = 3;
		if ( SpaceDimension == 2 ) AnglePartEnd = 1;
		for ( unsigned int i = 0; i < AnglePartEnd; i++ )
		{
			newscales[ i ] *= scaler;
		}

		/** And give it to the optimizer.*/
		m_Registration->GetAsITKBaseType()->GetOptimizer()->SetScales( newscales );
		
	} // end BeforeRegistration
	
	
	
	
	

} // end namespace elastix


#endif // end #ifndef __elxEulerTransform_HXX_

