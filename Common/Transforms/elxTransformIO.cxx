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

#include "xoutmain.h"

#include <itkTransformBase.h>
#include <itkTransformFactoryBase.h>
#include <itkTransformFileReader.h>
#include <itkTransformFileWriter.h>

#include <string>


std::string
elastix::TransformIO::ConvertITKNameOfClassToElastixClassName(const std::string & itkNameOfClass)
{
  // For example (ITK NameOfClass ==> elastix ClassName):
  //
  // "AffineTransform" ==> "AffineTransform"
  // "Euler2DTransform" ==> "EulerTransform"
  // "Similarity3DTransform" ==> "SimilarityTransform"

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
  // itk::TransformBase::GetNameOfClass() may yield a string like the following, for an elastix ITK transform:
  // - "AdvancedMatrixOffsetTransformBase"
  // - "AdvancedTranslationTransform"
  // - "SimilarityTransform"
  // - "EulerTransform"
  // - "AdvancedBSplineDeformableTransform"
  std::string name = elxTransform.GetNameOfClass();

  if (name == "AdvancedMatrixOffsetTransformBase")
  {
    name = "AffineTransform";
  }
  else
  {
    if (name == "AdvancedBSplineDeformableTransform")
    {
      name = "BSplineTransform";
    }
    else
    {
      const std::string advancedSubstring = "Advanced";

      if ((name.size() > advancedSubstring.size()) &&
          std::equal(name.cbegin(), name.cbegin() + advancedSubstring.size(), advancedSubstring.cbegin()))
      {
        name.erase(0, advancedSubstring.size());
      }
    }
  }

  const std::string transformSubstring = "Transform";
  const auto        transformSubstringPosition = name.find(transformSubstring);

  if (transformSubstringPosition == std::string::npos)
  {
    return nullptr;
  }
  const auto inputSpaceDimension = elxTransform.GetInputSpaceDimension();
  const auto outputSpaceDimension = elxTransform.GetOutputSpaceDimension();

  // For EulerTransform and SimilarityTransform, ITK has "2D"
  // or "3D" inserted in the class name.

  const auto substr = name.substr(0, transformSubstringPosition);
  const auto instanceName = (((substr == "Euler") || (substr == "Similarity"))
                               ? (substr + std::to_string(inputSpaceDimension) + 'D' + transformSubstring)
                               : name) +
                            "_double_" + std::to_string(inputSpaceDimension) + '_' +
                            std::to_string(outputSpaceDimension);

  // Initialize the factory.
  itk::TransformFactoryBase::GetFactory();

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
