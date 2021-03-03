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
#include "elxTransformIO.h"

#include "elxBaseComponent.h"
#include "elxConfiguration.h"

#include "xoutmain.h"

#include <itkTransformBase.h>
#include <itkTransformFactoryBase.h>
#include <itkTransformFileWriter.h>

#include <string>

itk::TransformBase::Pointer
elastix::TransformIO::CreateCorrespondingItkTransform(const elx::BaseComponent & elxTransform,
                                                      const unsigned             fixedImageDimension,
                                                      const unsigned             movingImageDimension)
{
  // Initialize the factory.
  itk::TransformFactoryBase::GetFactory();

  // The string returned by elxGetClassName() corresponds to ITK's GetNameOfClass(),
  // for AffineTransform, BSplineTransform, and TranslationTransform.
  // Note that for EulerTransform and SimilarityTransform, ITK has "2D"
  // or "3D" inserted in the class name.
  // For other transforms, the correspondence between elastix and ITK class names
  // appears less obvious.

  const std::string elxClassName = elxTransform.elxGetClassName();
  const std::string transformSubstring = "Transform";
  const auto        transformSubstringPosition = elxClassName.find(transformSubstring);

  if (transformSubstringPosition == std::string::npos)
  {
    return nullptr;
  }
  const auto substr = elxClassName.substr(0, transformSubstringPosition);
  const auto instanceName = (((substr == "Euler") || (substr == "Similarity"))
                               ? (substr + std::to_string(fixedImageDimension) + 'D' + transformSubstring)
                               : elxClassName) +
                            "_double_" + std::to_string(fixedImageDimension) + '_' +
                            std::to_string(movingImageDimension);
  const auto instance = itk::ObjectFactoryBase::CreateInstance(instanceName.c_str());
  return dynamic_cast<itk::TransformBase *>(instance.GetPointer());
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
