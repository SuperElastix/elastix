#ifndef __elxTranslationTransform_HXX_
#define __elxTranslationTransform_HXX_

#include "elxTranslationTransform.h"

namespace elastix
{
	using namespace itk;

	/**
	 * ********************* Constructor ****************************
	 */
	
	template <class TElastix>
		TranslationTransformElastix<TElastix>
		::TranslationTransformElastix()
	{
		this->m_TranslationTransform = 
			TranslationTransformType::New();
		this->SetCurrentTransform( this->m_TranslationTransform );
	} // end Constructor


	/*
	 * ******************* BeforeRegistration ***********************
	 */
	
	template <class TElastix>
		void TranslationTransformElastix<TElastix>
		::BeforeRegistration(void)
	{
		/** Give initial parameters to this->m_Registration.*/
		this->InitializeTransform();
		
	} // end BeforeRegistration
	

	/**
	 * ************************* InitializeTransform *********************
	 */

	template <class TElastix>
		void TranslationTransformElastix<TElastix>
		::InitializeTransform( void )
	{
		
		/** Set all parameters to zero (no translation */
		this->m_TranslationTransform->SetIdentity();
		
		/** Check if user wants automatic transform initialization; false by default. */
		std::string automaticTransformInitializationString("false");
		bool automaticTransformInitialization = false;
		this->m_Configuration->ReadParameter(
			automaticTransformInitializationString,
			"AutomaticTransformInitialization", 0);
		if ( (automaticTransformInitializationString == "true") &&
			(this->Superclass1::GetInitialTransform() == 0) )
		{
			automaticTransformInitialization = true;
		}

		/** 
		 * Run the itkTransformInitializer if:
		 *  the user asked for AutomaticTransformInitialization
		 */
		if ( automaticTransformInitialization ) 
		{
	    /** Use the TransformInitializer to determine an initial translation */
			TransformInitializerPointer transformInitializer = 
				TransformInitializerType::New();
			transformInitializer->SetFixedImage(
				this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
			transformInitializer->SetMovingImage(
				this->m_Registration->GetAsITKBaseType()->GetMovingImage() );
			transformInitializer->SetTransform(this->m_TranslationTransform);
			transformInitializer->GeometryOn();
			transformInitializer->InitializeTransform();
		}

		/** Set the initial parameters in this->m_Registration.*/
		this->m_Registration->GetAsITKBaseType()->
			SetInitialTransformParameters( this->GetParameters() );


	} // end InitializeTransform
	
	
} // end namespace elastix


#endif // end #ifndef __elxTranslationTransform_HXX_

