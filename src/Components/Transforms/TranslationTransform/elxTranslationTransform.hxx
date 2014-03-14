/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

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
