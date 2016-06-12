/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __elxTranslationTransform_HXX_
#define __elxTranslationTransform_HXX_

#include "elxTranslationTransform.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

template< class TElastix >
TranslationTransformElastix< TElastix >
::TranslationTransformElastix()
{
  this->m_TranslationTransform
    = TranslationTransformType::New();
  this->SetCurrentTransform( this->m_TranslationTransform );
}   // end Constructor


/*
 * ******************* BeforeRegistration ***********************
 */

template< class TElastix >
void
TranslationTransformElastix< TElastix >
::BeforeRegistration( void )
{
  /** Give initial parameters to this->m_Registration.*/
  this->InitializeTransform();

}   // end BeforeRegistration


/**
 * ************************* InitializeTransform *********************
 */

template< class TElastix >
void
TranslationTransformElastix< TElastix >
::InitializeTransform( void )
{

  /** Set all parameters to zero (no translation */
  this->m_TranslationTransform->SetIdentity();

  /** Check if user wants automatic transform initialization; false by default. */
  bool automaticTransformInitialization = false;
  bool tmpBool                          = false;
  this->m_Configuration->ReadParameter( tmpBool,
    "AutomaticTransformInitialization", 0 );
  if( tmpBool && this->Superclass1::GetInitialTransform() == 0 )
  {
    automaticTransformInitialization = true;
  }

  /**
   * Run the itkTransformInitializer if:
   *  the user asked for AutomaticTransformInitialization
   */
  if( automaticTransformInitialization )
  {
    /** Use the TransformInitializer to determine an initial translation */
    TransformInitializerPointer transformInitializer
      = TransformInitializerType::New();
    transformInitializer->SetFixedImage(
      this->m_Registration->GetAsITKBaseType()->GetFixedImage() );
    transformInitializer->SetMovingImage(
      this->m_Registration->GetAsITKBaseType()->GetMovingImage() );
    transformInitializer->SetFixedMask( this->GetElastix()->GetFixedMask() );
    transformInitializer->SetMovingMask( this->GetElastix()->GetMovingMask() );
    transformInitializer->SetTransform( this->m_TranslationTransform );

    /** Select the method of initialization. Default: "GeometricalCenter". */
    transformInitializer->GeometryOn();
    std::string method = "GeometricalCenter";
    this->m_Configuration->ReadParameter( method,
      "AutomaticTransformInitializationMethod", 0 );
    if( method == "CenterOfGravity" )
    {
      transformInitializer->MomentsOn();
    }

    transformInitializer->InitializeTransform();
  }

  /** Set the initial parameters in this->m_Registration.*/
  this->m_Registration->GetAsITKBaseType()->
  SetInitialTransformParameters( this->GetParameters() );

  /** Give feedback. */
  // \todo: should perhaps also print fixed parameters
  elxout << "Transform parameters are initialized as: "
         << this->GetParameters() << std::endl;

}   // end InitializeTransform


} // end namespace elastix

#endif // end #ifndef __elxTranslationTransform_HXX_
