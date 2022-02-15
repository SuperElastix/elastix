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
#ifndef elxBaseComponentSE_h
#define elxBaseComponentSE_h

#include "elxBaseComponent.h"
#include "elxConfiguration.h"

// ITK header files:
#include <itkMacro.h> // For ITK_DISALLOW_COPY_AND_MOVE.
#include <itkWeakPointer.h>

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

template <class TElastix>
class ITK_TEMPLATE_EXPORT BaseComponentSE : public BaseComponent
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BaseComponentSE);

  /** Standard stuff. */
  using Self = BaseComponentSE;
  using Superclass = BaseComponent;

  /** Elastix typedef. */
  using ElastixType = TElastix;

  /** Configuration pointer type. */
  using ConfigurationPointer = Configuration::Pointer;

  /** RegistrationType; NB: this is the elx::RegistrationBase
   * not an itk::Object or something like that.
   */
  using RegistrationType = typename ElastixType::RegistrationBaseType;

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
  void
  SetElastix(ElastixType * _arg);

  /** itkGetModifiableObjectMacro( Elastix, ElastixType );
   * without the itkDebug call.
   */
  ElastixType *
  GetElastix() const
  {
    return this->m_Elastix.GetPointer();
  }

  int
  RemoveTargetCellFromIterationInfo(const char * const name)
  {
    return this->m_Elastix->GetIterationInfo().xl::xoutrow::RemoveTargetCell(name);
  }

  xl::xoutbase &
  GetIterationInfoAt(const char * const name)
  {
    return this->m_Elastix->GetIterationInfoAt(name);
  }

  void
  AddTargetCellToIterationInfo(const char * const name)
  {
    return this->m_Elastix->AddTargetCellToIterationInfo(name);
  }

  /** itkGetModifiableObjectMacro(Configuration, Configuration);
   * The configuration object provides functionality to
   * read parameters and command line arguments.
   */
  Configuration *
  GetConfiguration() const
  {
    return this->m_Configuration.GetPointer();
  }


  /** Set the configuration. Added for transformix. */
  void
  SetConfiguration(Configuration * _arg);

  /** Get a pointer to the Registration component.
   * This is a convenience function, since the registration
   * component is needed often by other components.
   * It could be accessed also via GetElastix->GetElxRegistrationBase().
   */
  RegistrationType *
  GetRegistration() const
  {
    return this->m_Registration;
  }


protected:
  BaseComponentSE() = default;
  ~BaseComponentSE() override = default;

  itk::WeakPointer<TElastix> m_Elastix{};
  ConfigurationPointer       m_Configuration{};
  RegistrationType *         m_Registration{};

private:
  virtual const itk::Object &
  GetSelf() const = 0;

  virtual itk::Object &
  GetSelf() = 0;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxBaseComponentSE.hxx"
#endif

#endif // end #ifndef elxBaseComponentSE_h
