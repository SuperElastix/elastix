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
		/** Here is an heuristic rule for estimating good values for
		 * the rotation/translation scales.
		 *
		 * 1) Estimate the bounding box of your points (in physical units).
		 * 2) Take the 3D Diagonal of that bounding box
		 * 3) Multiply that by 10.0.
		 * 4) use 1.0 /[ value from (3) ] as the translation scaling value.
		 * 5) use 1.0 as the rotation scaling value.
		 *
		 * With this operation you bring the translation units
		 * to the range of rotations (e.g. around -1 to 1).
		 * After that, all your registration parameters are
		 * in the relaxed range of -1:1. At that point you
		 * can start setting your optimizer with step lengths
		 * in the ranges of 0.001 if you are conservative, or
		 * in the range of 0.1 if you want to live dangerously.
		 * (0.1 radians is about 5.7 degrees).
		 * 
		 * This heuristic rule is based on the naive assumption
		 * that your registration may require translations as
		 * large as 1/10 of the diagonal of the bounding box.
		 */
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

