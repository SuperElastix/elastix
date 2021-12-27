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
#ifndef elxBaseComponent_h
#define elxBaseComponent_h

/**
 * ******************* elxBaseComponent.h *************************
 *
 * This file defines the class elx::BaseComponent, from which all
 * elastix components should inherit. It contains some methods that
 * each component is supposed to have.
 *
 * The namespace alias elx is defined in this file.
 */

#include "itkMacro.h" // itkTypeMacroNoParent
#include <string>

/** All elastix components should be in namespace elastix. */
namespace elastix
{

/**
 * \class BaseComponent
 *
 * \brief The BaseComponent class is a class that all elastix
 * components should inherit from.
 *
 * Most elastix component inherit both from an ITK class and
 * from the elx::BaseComponent class. The BaseComponent class
 * contains some methods that each component is supposed
 * to have, but are not defined in itk::Object.
 *
 * \sa BaseComponentSE
 * \ingroup Install
 */

class BaseComponent
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BaseComponent);

  /**
   * Callback methods that each component of elastix is supposed
   * to have. These methods can be overridden in each base component
   * (like MetricBase, TransformBase, etc.). In this way similar
   * behavior for a group of components can be implemented.
   */
  virtual int
  BeforeAllBase()
  {
    return 0;
  }
  virtual int
  BeforeAll()
  {
    return 0;
  }

  /**
   * Callback methods that each component of elastix is supposed
   * to have. These methods can be overridden in each base component
   * (like MetricBase, TransformBase, etc.). In this way similar
   * behavior for a group of components can be implemented.
   */
  virtual void
  BeforeRegistrationBase()
  {}
  virtual void
  BeforeEachResolutionBase()
  {}
  virtual void
  AfterEachResolutionBase()
  {}
  virtual void
  AfterEachIterationBase()
  {}
  virtual void
  AfterRegistrationBase()
  {}

  /**
   * Callback methods that each component of elastix is supposed
   * to have. These methods can be overridden in each single
   * component (like MattesMutualInformationMetric) to achieve
   * behavior, specific for that single component.
   */
  virtual void
  BeforeRegistration()
  {}
  virtual void
  BeforeEachResolution()
  {}
  virtual void
  AfterEachResolution()
  {}
  virtual void
  AfterEachIteration()
  {}
  virtual void
  AfterRegistration()
  {}

  /**
   * The name of the component in the ComponentDatabase.
   * Override this function not directly, but with the
   * elxClassNameMacro("name").
   */
  virtual const char *
  elxGetClassName() const;

  itkTypeMacroNoParent(BaseComponent);

  /** Set the component label, which consists of a label
   * ( "Metric", "Transform") and an index number. In case
   * more metrics are used simultaneously each metric will have
   * its own index number. This can be used when reading the
   * parameter file for example, to distinguish between options
   * meant for Metric0 and for Metric1.
   */
  void
  SetComponentLabel(const char * label, unsigned int idx);

  /** Get the componentlabel as a string: "Metric0" for example. */
  const char *
  GetComponentLabel() const;

  static bool
  IsElastixLibrary();

  static void
  InitializeElastixExecutable();

  /** Similar to `dynamic_cast<ITKBaseType*>(baseComponent)`, but without
   * using RTTI (run-time type information).
   */
  template <typename TBaseComponent>
  static auto
  AsITKBaseType(TBaseComponent * const baseComponent) -> decltype(baseComponent->GetAsITKBaseType())
  {
    return (baseComponent == nullptr) ? nullptr : baseComponent->GetAsITKBaseType();
  }

protected:
  BaseComponent() = default;
  virtual ~BaseComponent() = default;

private:
  std::string m_ComponentLabel;
};

} // end namespace elastix

/** Define an alias for the elastix namespace.*/
namespace elx = elastix;

#endif // end #ifndef elxBaseComponent_h
