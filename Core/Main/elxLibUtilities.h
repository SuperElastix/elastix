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
#ifndef elxLibUtilities_h
#define elxLibUtilities_h

#include "elxForEachSupportedImageType.h"

#include <itkCastImageFilter.h>
#include <itkDataObject.h>
#include <itkSmartPointer.h>

#include <map>
#include <string>
#include <type_traits> // For is_same_v.
#include <vector>


namespace elastix::LibUtilities
{
using ParameterValuesType = std::vector<std::string>;
using ParameterMapType = std::map<std::string, ParameterValuesType>;


/** Sets the specified parameter value. Warns when it overrides existing parameter values. */
void
SetParameterValueAndWarnOnOverride(ParameterMapType &  parameterMap,
                                   const std::string & parameterName,
                                   const std::string & parameterValue);


/** Retrieves the PixelType string value of the specified parameter. Returns "float" by default. */
std::string
RetrievePixelTypeParameterValue(const ParameterMapType & parameterMap, const std::string & parameterName);

template <typename TInputImage>
itk::SmartPointer<itk::DataObject>
CastToInternalPixelType(itk::SmartPointer<TInputImage> inputImage, const std::string & internalPixelTypeString)
{
  if (inputImage == nullptr)
  {
    itkGenericExceptionMacro("The specified input image should not be null!");
  }

  itk::SmartPointer<itk::DataObject> outputImage;

  elx::ForEachSupportedImageTypeUntilTrue([inputImage, &outputImage, &internalPixelTypeString](const auto elxTypedef) {
    using ElxTypedef = decltype(elxTypedef);

    if constexpr (TInputImage::ImageDimension == ElxTypedef::MovingDimension)
    {
      using InternalImageType = typename ElxTypedef::MovingImageType;

      if (internalPixelTypeString == ElxTypedef::MovingPixelTypeString)
      {
        if constexpr (std::is_same_v<TInputImage, InternalImageType>)
        {
          outputImage = inputImage;
        }
        else
        {
          const auto castFilter = itk::CastImageFilter<TInputImage, InternalImageType>::New();
          castFilter->SetInput(inputImage);
          castFilter->Update();
          outputImage = castFilter->GetOutput();
        }
        return true;
      }
    }
    return false;
  });

  if (outputImage == nullptr)
  {
    itkGenericExceptionMacro("Failed to cast to the specified internal pixel type \""
                             << internalPixelTypeString
                             << "\". It may need to be added to the CMake variable ELASTIX_IMAGE_"
                             << TInputImage::ImageDimension << "D_PIXELTYPES.");
  }
  return outputImage;
}

} // namespace elastix::LibUtilities

#endif
