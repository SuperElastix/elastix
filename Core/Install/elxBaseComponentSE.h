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
#ifndef __elxBaseComponentSE_h
#define __elxBaseComponentSE_h

#include "elxBaseComponent.h"
#include "itkMacro.h" // For ITK_DISALLOW_COPY_AND_ASSIGN.
#include "itkWeakPointer.h"

namespace elastix
{

/**
 * \class BaseComponentSE
 *
 * \brief The BaseComponentSE class is a base class for elastix
 * components that provides some basic functionality.
 *
 * Most elastix component will not directly inherit from the
 * elx::BaseComponent class but from this one, since it adds
 * some methods that most methods need anyway, such as
 * Set/GetElastix, Set/GetConfiguration.
 *
 * \sa BaseComponent
 * \ingroup Install
 */

template< class TElastix >
class BaseComponentSE : public BaseComponent
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(BaseComponentSE);

  /** Standard stuff. */
  typedef BaseComponentSE Self;
  typedef BaseComponent   Superclass;

  /** Elastix typedef's. */
  typedef TElastix                        ElastixType;
  typedef itk::WeakPointer< ElastixType > ElastixPointer;

  /** ConfigurationType. */
  typedef Configuration          ConfigurationType;
  typedef Configuration::Pointer ConfigurationPointer;

  /** RegistrationType; NB: this is the elx::RegistrationBase
   * not an itk::Object or something like that.
   */
  typedef typename ElastixType::RegistrationBaseType RegistrationType;
  typedef RegistrationType *                         RegistrationPointer;

  /**
   * Get/Set functions for Elastix.
   * The Set-functions cannot be defined with the itkSetObjectMacro,
   * since this class does not derive from itk::Object and
   * thus does not have a Modified() method.
   *
   * This method checks if this instance of the class can be casted
   * (dynamically) to an itk::Object. If yes, it calls Modified()
   *
   * Besides setting m_Elastix, this method also sets m_Configuration
   * and m_Registration.
   */
  void SetElastix( ElastixType * _arg );

  /** itkGetModifiableObjectMacro( Elastix, ElastixType );
   * without the itkDebug call.
   */
  ElastixType * GetElastix( void ) const
  {
    return this->m_Elastix.GetPointer();
  }


  /** itkGetModifiableObjectMacro(Configuration, ConfigurationType);
   * The configuration object provides functionality to
   * read parameters and command line arguments.
   */
  ConfigurationType * GetConfiguration( void ) const
  {
    return this->m_Configuration.GetPointer();
  }


  /** Set the configuration. Added for transformix. */
  void SetConfiguration( ConfigurationType * _arg );

  /** Get a pointer to the Registration component.
   * This is a convenience function, since the registration
   * component is needed often by other components.
   * It could be accessed also via GetElastix->GetElxRegistrationBase().
   */
  RegistrationPointer GetRegistration( void ) const
  {
    return this->m_Registration;
  }


protected:

  BaseComponentSE() = default;
  ~BaseComponentSE() override = default;

  ElastixPointer       m_Elastix{};
  ConfigurationPointer m_Configuration{};
  RegistrationPointer  m_Registration{};
};

} //end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#include "elxBaseComponentSE.hxx"
#endif

#endif // end #ifndef __elxBaseComponentSE_h
