#ifndef __elxTranslationTransform_HXX_
#define __elxTranslationTransform_HXX_

#include "elxTranslationTransform.h"

namespace elastix
{
	using namespace itk;

	
	/*
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		TranslationTransformElastix<TElastix>
		::TranslationTransformElastix()
	{
	} // end Constructor
	
	
	/*
	 * ******************* BeforeRegistration ***********************
	 */
	
	template <class TElastix>
		void TranslationTransformElastix<TElastix>
		::BeforeRegistration(void)
	{
		/** Give initial parameters to this->m_Registration.*/
		ParametersType dummyInitialParameters( this->GetNumberOfParameters() );
		dummyInitialParameters.Fill(0.0);
		
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParameters( dummyInitialParameters );
		
	} // end BeforeRegistration
	
	
	
} // end namespace elastix


#endif // end #ifndef __elxTranslationTransform_HXX_

