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
 *
 * Some header files are included that most components need.
 */

/** Get rid of warnings about too long variable names. */
#ifdef _MSC_VER
#  pragma warning(disable : 4786)
#  pragma warning(disable : 4503)
#endif

#include "itkMacro.h" // itkTypeMacroNoParent
#include "itkMatrix.h"

#include <vnl_vector.h>

#include <string>
#include <type_traits> // For is_integral and is_same.
#include <vector>

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
  ITK_DISALLOW_COPY_AND_ASSIGN(BaseComponent);

  /**
   * Callback methods that each component of elastix is supposed
   * to have. These methods can be overridden in each base component
   * (like MetricBase, TransformBase, etc.). In this way similar
   * behavior for a group of components can be implemented.
   */
  virtual int
  BeforeAllBase(void)
  {
    return 0;
  }
  virtual int
  BeforeAll(void)
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
  BeforeRegistrationBase(void)
  {}
  virtual void
  BeforeEachResolutionBase(void)
  {}
  virtual void
  AfterEachResolutionBase(void)
  {}
  virtual void
  AfterEachIterationBase(void)
  {}
  virtual void
  AfterRegistrationBase(void)
  {}

  /**
   * Callback methods that each component of elastix is supposed
   * to have. These methods can be overridden in each single
   * component (like MattesMutualInformationMetric) to achieve
   * behavior, specific for that single component.
   */
  virtual void
  BeforeRegistration(void)
  {}
  virtual void
  BeforeEachResolution(void)
  {}
  virtual void
  AfterEachResolution(void)
  {}
  virtual void
  AfterEachIteration(void)
  {}
  virtual void
  AfterRegistration(void)
  {}

  /**
   * The name of the component in the ComponentDatabase.
   * Override this function not directly, but with the
   * elxClassNameMacro("name").
   */
  virtual const char *
  elxGetClassName(void) const;

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
  GetComponentLabel(void) const;

  static bool
  IsElastixLibrary();

  static void
  InitializeElastixExecutable();


  /** Overload set, similar to C++17 `std::size(const TContainer&)` (which can only be
   * used within the implementation of elastix is upgraded to C++17 or higher).
   */
  template <typename TContainer, unsigned NDimension = TContainer::Dimension>
  static std::size_t
  GetNumberOfElements(const TContainer &)
  {
    return NDimension;
  }

  template <typename TValue>
  static std::size_t
  GetNumberOfElements(const vnl_vector<TValue> & vnlVector)
  {
    return vnlVector.size();
  }


  /** Convenience function to convert seconds to day, hour, minute, second format. */
  static std::string
  ConvertSecondsToDHMS(const double totalSeconds, const unsigned int precision);

  /** Convenience function to convert a boolean to a text string. */
  static constexpr const char *
  BoolToString(const bool arg)
  {
    return arg ? "true" : "false";
  }

  /** Convenience function overload to convert a Boolean to a text string. */
  static std::string
  ToString(const bool arg)
  {
    return BoolToString(arg);
  }

  /** Convenience function overload to convert a floating point to a text string. */
  static std::string
  ToString(const double scalar)
  {
    std::ostringstream stringStream;
    stringStream << scalar;
    return stringStream.str();
  }

  /** Convenience function overload to convert an integer to a text string. */
  template <typename TScalarValue>
  static std::string
  ToString(const TScalarValue scalar)
  {
    static_assert(std::is_integral<TScalarValue>::value, "An integer type expected!");
    static_assert(!std::is_same<TScalarValue, bool>::value, "No bool expected!");
    return std::to_string(scalar);
  }


  /** Convenience function overload to convert a container to a vector of
   * text strings. The container may be an itk::Size, itk::Index,
   * itk::Point<double,N>, or itk::Vector<double,N>, or
   * itk::OptimizationParameters<double>.
   *
   * The C++ SFINAE idiom is being used to ensure that the argument type
   * supports standard C++ iteration.
   */
  template <typename TContainer, typename SFINAE = typename TContainer::iterator>
  static std::vector<std::string>
  ToVectorOfStrings(const TContainer & container)
  {
    std::vector<std::string> result;

    // Note: Uses TContainer::Dimension instead of container.size(),
    // because itk::FixedArray::size() is not yet included with ITK 5.1.1.
    result.reserve(GetNumberOfElements(container));

    for (const auto element : container)
    {
      result.push_back(BaseComponent::ToString(element));
    }
    return result;
  }

  /** Convenience function overload to convert a 2-D matrix to a vector of
   * text strings. Typically used for an itk::ImageBase::DirectionType.
   */
  template <typename T, unsigned int NRows, unsigned int NColumns>
  static std::vector<std::string>
  ToVectorOfStrings(const itk::Matrix<T, NRows, NColumns> & matrix)
  {
    std::vector<std::string> result;
    result.reserve(NColumns * NRows);

    for (unsigned column{}; column < NColumns; ++column)
    {
      for (unsigned row{}; row < NRows; ++row)
      {
        result.push_back(BaseComponent::ToString(matrix(row, column)));
      }
    }
    return result;
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
