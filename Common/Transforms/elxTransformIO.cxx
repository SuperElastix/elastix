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

// Avoid creation of instances of `itk::ImageIOFactoryRegisterManager`, `itk::MeshIOFactoryRegisterManager`, and
// `itk::TransformIOFactoryRegisterManager` at this point.
#undef ITK_IO_FACTORY_REGISTER_MANAGER

#include "elxTransformIO.h"

#include "elxBaseComponent.h"
#include "elxConfiguration.h"
#include "elxSupportedImageDimensions.h"

#include "xoutmain.h"

#include "itkAdvancedBSplineDeformableTransformBase.h"

#include <itkTransformBase.h>
#include <itkTransformFactoryBase.h>
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>

#include <string>


namespace
{

// Returns the spline order of the transform. Returns zero if the transform has no spline order, of if its spline order
// has a "default value" of 3.
template <std::size_t NDimension>
unsigned
GetSplineOrderFromBSplineDeformableTransform(const itk::TransformBase & elxTransform)
{
  const auto bSplineDeformableTransform =
    dynamic_cast<const itk::AdvancedBSplineDeformableTransformBase<double, NDimension> *>(&elxTransform);

  if (bSplineDeformableTransform == nullptr)
  {
    return 0;
  }
  const auto     splineOrder = bSplineDeformableTransform->GetSplineOrder();
  constexpr auto defaultSplineOrder = 3;

  return (splineOrder == defaultSplineOrder) ? 0 : bSplineDeformableTransform->GetSplineOrder();
}


template <std::size_t... NDimension>
unsigned
GetOptionalSplineOrderByImageDimensionSequence(const itk::TransformBase & elxTransform,
                                               const std::index_sequence<NDimension...>)
{
  const unsigned splineOrders[] = { GetSplineOrderFromBSplineDeformableTransform<NDimension>(elxTransform)... };
  return *std::max_element(std::cbegin(splineOrders), std::cend(splineOrders));
}

} // namespace


std::string
elastix::TransformIO::ConvertITKNameOfClassToElastixClassName(const std::string & itkNameOfClass)
{
  // For example (ITK NameOfClass ==> elastix ClassName):
  //
  // "AffineTransform" ==> "AffineTransform"
  // "Euler2DTransform" ==> "EulerTransform"
  // "Similarity3DTransform" ==> "SimilarityTransform"

  if (itkNameOfClass == "BSplineTransform")
  {
    // The elastix "RecursiveBSplineTransform" is faster than the elastix "BSplineTransform".
    return "RecursiveBSplineTransform";
  }

  auto name = itkNameOfClass;

  // Remove "nD" from ITK's "Euler2DTransform", "Similarity3DTransform", etc.
  const auto found = std::min(name.find("2D"), name.find("3D"));

  if (found != std::string::npos)
  {
    name.erase(found, 2);
  }
  return name;
}


itk::TransformBase::Pointer
elastix::TransformIO::ConvertItkTransformBaseToSingleItkTransform(const itk::TransformBase & elxTransform)
{
  const std::string className = [&elxTransform]() -> std::string {
    // itk::TransformBase::GetNameOfClass() may yield a string like the following, for an elastix ITK transform:
    // - "AdvancedMatrixOffsetTransformBase"
    // - "AdvancedTranslationTransform"
    // - "SimilarityTransform"
    // - "EulerTransform"
    // - "AdvancedBSplineDeformableTransform"
    const std::string name = elxTransform.GetNameOfClass();

    if (name == "AdvancedMatrixOffsetTransformBase")
    {
      return "AffineTransform";
    }
    if (name == "AdvancedBSplineDeformableTransform" || name == "RecursiveBSplineTransform")
    {
      return "BSplineTransform";
    }

    const std::string transformSubstring = "Transform";

    if (name.size() > transformSubstring.size())
    {
      const auto transformSubstringPosition = name.size() - transformSubstring.size();

      if (std::equal(name.cbegin() + transformSubstringPosition, name.cend(), transformSubstring.cbegin()))
      {
        const std::string advancedSubstring = "Advanced";

        if (std::equal(name.cbegin(),
                       name.cbegin() + advancedSubstring.size(),
                       advancedSubstring.cbegin(),
                       advancedSubstring.cend()))
        {
          // Just chop off the "Advanced" substring.
          return std::string(name.c_str() + advancedSubstring.size());
        }

        const auto substr = name.substr(0, transformSubstringPosition);

        if ((substr == "Euler") || (substr == "Similarity"))
        {
          // For EulerTransform and SimilarityTransform, ITK has "2D" or "3D" inserted in the class name.
          return substr + std::to_string(elxTransform.GetInputSpaceDimension()) + 'D' + transformSubstring;
        }
      }
    }

    return "";
  }();

  if (className.empty())
  {
    return nullptr;
  }

  // When the transform has a non-zero non-default SplineOrder, it must be appended to the instance name. Specifically
  // relevant for b-spline transform types (having instance names like "BSplineTransform_double_2_2_2" and .
  const auto optionalSplineOrder =
    GetOptionalSplineOrderByImageDimensionSequence(elxTransform, SupportedFixedImageDimensionSequence);
  const auto optionalSplineOrderPostfix =
    (optionalSplineOrder == 0) ? std::string() : ('_' + std::to_string(optionalSplineOrder));

  // Initialize the factory.
  itk::TransformFactoryBase::GetFactory();

  const auto instanceName = className + "_double_" + std::to_string(elxTransform.GetInputSpaceDimension()) + '_' +
                            std::to_string(elxTransform.GetOutputSpaceDimension()) + optionalSplineOrderPostfix;
  const auto instance = itk::ObjectFactoryBase::CreateInstance(instanceName.c_str());
  const auto itkTransform = dynamic_cast<itk::TransformBase *>(instance.GetPointer());
  if (itkTransform == nullptr)
  {
    return nullptr;
  }
  itkTransform->SetFixedParameters(elxTransform.GetFixedParameters());
  itkTransform->SetParameters(elxTransform.GetParameters());
  return itkTransform;
}


void
elastix::TransformIO::Write(const itk::TransformBase & itkTransform, const std::string & fileName)
{
  try
  {
    const auto writer = itk::TransformFileWriter::New();

    writer->SetInput(&itkTransform);
    writer->SetFileName(fileName);
    writer->Update();
  }
  catch (const std::exception & stdException)
  {
    xl::xout["error"] << "Error trying to write " << fileName << ":\n" << stdException.what() << std::endl;
  }
}


itk::SmartPointer<itk::TransformBase>
elastix::TransformIO::Read(const std::string & fileName)
{
  const auto reader = itk::TransformFileReader::New();

  reader->SetFileName(fileName);
  reader->Update();

  const auto transformList = reader->GetModifiableTransformList();
  assert(transformList != nullptr);

  // More than one transform is not yet supported.
  assert(transformList->size() <= 1);

  return transformList->empty() ? nullptr : transformList->front();
}


std::string
elastix::TransformIO::MakeDeformationFieldFileName(Configuration &     configuration,
                                                   const std::string & transformParameterFileName)
{
  // Get the last part of the filename of the transformParameter-file,
  // which is going to be part of the filename of the deformationField image.
  const std::string            transformParameterBaseName = "TransformParameters";
  const auto                   transformParameterBaseNameSize = transformParameterBaseName.size();
  const std::string::size_type pos = transformParameterFileName.rfind(transformParameterBaseName + '.');
  const std::string            lastpart =
    (pos == std::string::npos)
      ? ""
      : transformParameterFileName.substr(pos + transformParameterBaseNameSize,
                                          transformParameterFileName.size() - pos - transformParameterBaseNameSize - 4);

  // Create the filename of the deformationField image.
  std::string resultImageFormat = "mhd";
  configuration.ReadParameter(resultImageFormat, "ResultImageFormat", 0, false);

  return configuration.GetCommandLineArgument("-out") + "DeformationFieldImage" + lastpart + "." + resultImageFormat;
}
