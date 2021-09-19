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
#ifndef elxTransformIO_h
#define elxTransformIO_h

#include "itkAdvancedCombinationTransform.h"

#include <itkCompositeTransform.h>
#include <itkTransform.h>
#include <itkTransformBase.h>

#include <cassert>
#include <string>

namespace elastix
{
class BaseComponent;
class Configuration;

class TransformIO
{
public:
  static itk::OptimizerParameters<double>
  GetParameters(const bool fixed, const itk::TransformBase & transform)
  {
    return fixed ? transform.GetFixedParameters() : transform.GetParameters();
  }

  static void
  SetParameters(const bool fixed, itk::TransformBase & transform, const itk::OptimizerParameters<double> & parameters)
  {
    fixed ? transform.SetFixedParameters(parameters) : transform.SetParameters(parameters);
  }


  /// Converts the name of an ITK Transform class (as returned by
  /// `GetNameOfClass()`) to the corresponding elastix class name
  /// (as returned by `elxGetClassName()`).
  static std::string
  ConvertITKNameOfClassToElastixClassName(const std::string & itkNameOfClass);


  /// Converts the specified combination transform from elastix to the corresponding ITK composite transform. Returns
  /// null when the combination transform does not use composition.
  template <unsigned NDimension>
  static itk::SmartPointer<itk::CompositeTransform<double, NDimension>>
  ConvertToItkCompositeTransform(
    const itk::AdvancedCombinationTransform<double, NDimension> & advancedCombinationTransform)
  {
    const auto numberOfTransforms = advancedCombinationTransform.GetNumberOfTransforms();

    if ((numberOfTransforms > 1) && (!advancedCombinationTransform.GetUseComposition()))
    {
      // A combination of multiple transforms can only be converted to CompositeTransform when the original combination
      // uses composition.
      return nullptr;
    }

    const auto compositeTransform = itk::CompositeTransform<double, NDimension>::New();

    for (itk::SizeValueType n{}; n < numberOfTransforms; ++n)
    {
      const auto nthTransform = advancedCombinationTransform.GetNthTransform(n);
      const auto singleItkTransform = ConvertToSingleItkTransform(*nthTransform);
      compositeTransform->AddTransform((singleItkTransform == nullptr) ? nthTransform : singleItkTransform);
    }
    return compositeTransform;
  }


  /// Converts the specified single transform from elastix to the corresponding ITK transform. Returns null when ITK has
  /// no transform type that corresponds with this elastix transform.
  template <unsigned NDimension>
  static itk::SmartPointer<itk::Transform<double, NDimension, NDimension>>
  ConvertToSingleItkTransform(const itk::Transform<double, NDimension, NDimension> & elxTransform)
  {
    // Do not use this function for elastix combination transforms!
    using CombinationTransformType = itk::AdvancedCombinationTransform<double, NDimension>;
    assert(dynamic_cast<const CombinationTransformType *>(&elxTransform) == nullptr);

    return dynamic_cast<itk::Transform<double, NDimension, NDimension> *>(
      ConvertItkTransformBaseToSingleItkTransform(elxTransform).GetPointer());
  }


  static void
  Write(const itk::TransformBase & itkTransform, const std::string & fileName);

  static itk::TransformBase::Pointer
  Read(const std::string & fileName);

  /// Makes the deformation field file name, as used by BSplineTransformWithDiffusion and DeformationFieldTransform.
  template <typename TElastixTransform>
  static std::string
  MakeDeformationFieldFileName(const TElastixTransform & elxTransform)
  {
    return MakeDeformationFieldFileName(*(elxTransform.GetConfiguration()),
                                        elxTransform.GetElastix()->GetCurrentTransformParameterFileName());
  }

private:
  static itk::TransformBase::Pointer
  ConvertItkTransformBaseToSingleItkTransform(const itk::TransformBase & elxTransform);

  static std::string
  MakeDeformationFieldFileName(Configuration & configuration, const std::string & transformParameterFileName);
};
} // namespace elastix

#endif
