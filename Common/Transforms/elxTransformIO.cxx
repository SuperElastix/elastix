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

#include "xoutmain.h"

#include <itkTransformBase.h>
#include <itkTransformFactoryBase.h>
#include <itkTransformFileWriter.h>

#include <string>

itk::TransformBaseTemplate<double>::Pointer
elastix::TransformIO::CreateCorrespondingItkTransform(const elx::BaseComponent & elxTransform,
                                                      const unsigned                             fixedImageDimension,
                                                      const unsigned                             movingImageDimension)
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
  return dynamic_cast<itk::TransformBaseTemplate<double> *>(instance.GetPointer());
}

void
elastix::TransformIO::Write(const itk::TransformBaseTemplate<double> & itkTransform, const std::string & fileName)
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
