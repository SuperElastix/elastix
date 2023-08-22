/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
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
#ifndef itkElastixRegistrationMethod_hxx
#define itkElastixRegistrationMethod_hxx

#include "elxPixelTypeToString.h"
#include "itkElastixRegistrationMethod.h"

#include "elxLibUtilities.h"

#include <itkTransformFileWriter.h>

#include <algorithm> // For find.
#include <memory>    // For unique_ptr.

namespace itk
{

template <typename TFixedImage, typename TMovingImage>
ElastixRegistrationMethod<TFixedImage, TMovingImage>::ElastixRegistrationMethod()
{
  this->SetPrimaryInputName("FixedImage");
  this->SetNumberOfIndexedOutputs(2);

  this->AddRequiredInputName("MovingImage", 1);
  this->AddRequiredInputName("ParameterObject", 2);

  ParameterObjectPointer defaultParameterObject = elx::ParameterObject::New();
  defaultParameterObject->AddParameterMap(elx::ParameterObject::GetDefaultParameterMap("translation"));
  defaultParameterObject->AddParameterMap(elx::ParameterObject::GetDefaultParameterMap("affine"));
  defaultParameterObject->AddParameterMap(elx::ParameterObject::GetDefaultParameterMap("bspline"));
  defaultParameterObject->SetParameter("FixedInternalImagePixelType", "float");
#ifdef ELASTIX_USE_OPENCL
  defaultParameterObject->SetParameter("Resampler", "OpenCLResampler");
  defaultParameterObject->SetParameter("OpenCLResamplerUseOpenCL", "true");
  // Requires copius amounts of GPU memory
  // defaultParameterObject->SetParameter( "FixedImagePyramid", "OpenCLFixedGenericImagePyramid" );
  // defaultParameterObject->SetParameter( "OpenCLFixedGenericImagePyramidUseOpenCL", "true" );
  // defaultParameterObject->SetParameter( "MovingImagePyramid", "OpenCLMovingGenericImagePyramid" );
  // defaultParameterObject->SetParameter( "OpenCLMovingGenericImagePyramidUseOpenCL", "true" );
#endif
  this->SetParameterObject(defaultParameterObject);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GenerateData()
{
  using elx::LibUtilities::CastToInternalPixelType;
  using elx::LibUtilities::RetrievePixelTypeParameterValue;
  using elx::LibUtilities::SetParameterValueAndWarnOnOverride;

  DataObjectContainerPointer fixedMaskContainer = nullptr;
  DataObjectContainerPointer movingMaskContainer = nullptr;
  DataObjectContainerPointer resultImageContainer = nullptr;
  ElastixMainObjectPointer   transform = nullptr;
  FlatDirectionCosinesType   fixedImageOriginalDirection;

  // Split inputs into separate containers
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType("FixedMask", inputName))
    {
      if (fixedMaskContainer.IsNull())
      {
        fixedMaskContainer = DataObjectContainerType::New();
      }

      fixedMaskContainer->push_back(this->ProcessObject::GetInput(inputName));
      continue;
    }

    if (this->IsInputOfType("MovingMask", inputName))
    {
      if (movingMaskContainer.IsNull())
      {
        movingMaskContainer = DataObjectContainerType::New();
      }

      movingMaskContainer->push_back(this->ProcessObject::GetInput(inputName));
    }
  }

  // Set ParameterMap
  ParameterObjectPointer parameterObject =
    itkDynamicCastInDebugMode<elx::ParameterObject *>(this->ProcessObject::GetInput("ParameterObject"));
  ParameterMapVectorType parameterMapVector = parameterObject->GetParameterMaps();

  if (parameterMapVector.empty())
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  // Setup argument map
  ArgumentMapType argumentMap;

  const auto insertIfNotEmpty = [&argumentMap](const char * const argument, const std::string & fileName) {
    if (!fileName.empty())
    {
      argumentMap.insert(ArgumentMapEntryType(argument, fileName));
    }
  };

  insertIfNotEmpty("-t0", m_InitialTransformParameterFileName);
  insertIfNotEmpty("-fp", m_FixedPointSetFileName);
  insertIfNotEmpty("-mp", m_MovingPointSetFileName);

  // Setup output directory
  if (m_OutputDirectory.empty())
  {
    if (m_LogToFile)
    {
      itkExceptionMacro("LogToFileOn() requires an output directory to be specified.")
    }
  }
  else
  {
    if (!itksys::SystemTools::FileExists(m_OutputDirectory))
    {
      itkExceptionMacro("Output directory \"" << m_OutputDirectory << "\" does not exist.");
    }

    if (m_OutputDirectory.back() != '/' && m_OutputDirectory.back() != '\\')
    {
      this->SetOutputDirectory(m_OutputDirectory + "/");
    }

    argumentMap.insert(ArgumentMapEntryType("-out", m_OutputDirectory));
  }

  // Setup log file
  const std::string logFileName = m_OutputDirectory + (m_LogFileName.empty() ? "elastix.log" : m_LogFileName);

  // Set Number of threads
  if (m_NumberOfThreads > 0)
  {
    argumentMap.insert(ArgumentMapEntryType("-threads", std::to_string(m_NumberOfThreads)));
  }

  // Setup logging.
  const elx::log::guard logGuard(logFileName,
                                 m_EnableOutput && m_LogToFile,
                                 m_EnableOutput && m_LogToConsole,
                                 static_cast<elastix::log::level>(m_LogLevel));

  const auto getInitialTransformParameterMaps = [this]() -> ParameterMapVectorType {
    if (m_InitialTransformParameterObject)
    {
      return m_InitialTransformParameterObject->GetParameterMaps();
    }

    if (m_InitialTransform)
    {
      const auto transformToMap = [](const itk::TransformBase & transform) {
        return ParameterMapType{
          { "ITKTransformFixedParameters", elx::Conversion::ToVectorOfStrings(transform.GetFixedParameters()) },
          { "ITKTransformParameters", elx::Conversion::ToVectorOfStrings(transform.GetParameters()) },
          { "ITKTransformType", { transform.GetTransformTypeAsString() } },
          { "Transform", { elx::TransformIO::ConvertITKNameOfClassToElastixClassName(transform.GetNameOfClass()) } }
        };
      };

      const auto compositeTransform =
        dynamic_cast<const CompositeTransform<double, MovingImageDimension> *>(&*m_InitialTransform);

      if (compositeTransform)
      {
        const auto & transformQueue = compositeTransform->GetTransformQueue();

        ParameterMapVectorType transformParameterMaps(transformQueue.size());

        auto reverseIterator = transformParameterMaps.rbegin();

        for (const auto & transform : transformQueue)
        {
          if (transform == nullptr)
          {
            itkGenericExceptionMacro("One of the subtransforms of the specified composite transform is null!");
          }
          *reverseIterator = transformToMap(*transform);
          ++reverseIterator;
        }
        return transformParameterMaps;
      }

      // Assume in this case that it is just a single transform.
      assert((dynamic_cast<const MultiTransform<double, MovingImageDimension> *>(&*m_InitialTransform)) == nullptr);

      // For a single transform, there should be only a single transform parameter map.
      return ParameterMapVectorType{ transformToMap(*m_InitialTransform) };
    }
    if (m_ExternalInitialTransform)
    {
      return ParameterMapVectorType{ ParameterMapType{
        { "NumberOfParameters", { "0" } },
        { "Transform", { "ExternalTransform" } },
        { "TransformAddress", { elx::Conversion::ObjectPtrToString(m_ExternalInitialTransform) } } } };
    }
    return {};
  };

  ParameterMapVectorType transformParameterMapVector = getInitialTransformParameterMaps();

  if (!transformParameterMapVector.empty() && !m_OutputDirectory.empty())
  {
    std::string initialTransformParameterFileName = "NoInitialTransform";

    // Write InitialTransformParameters.0.txt, InitialTransformParameters.1.txt, InitialTransformParameters.2.txt, etc.
    unsigned i{};

    const auto & firstParameterMap = parameterMapVector.front();
    const auto   outputFileNameExtensionFound = firstParameterMap.find("ITKTransformOutputFileNameExtension");

    // Use the ITK TFM file format by default, when writing external transforms.
    const std::string outputFileNameExtension =
      (outputFileNameExtensionFound == firstParameterMap.end() || outputFileNameExtensionFound->second.empty())
        ? "tfm"
        : outputFileNameExtensionFound->second.front();

    for (auto transformParameterMap : transformParameterMapVector)
    {
      transformParameterMap["InitialTransformParameterFileName"] = { initialTransformParameterFileName };

      if (const auto transformFound = transformParameterMap.find("Transform");
          transformFound != transformParameterMap.end() &&
          transformFound->second == ParameterValueVectorType{ "ExternalTransform" })
      {
        Object * externalTransform{};

        // Retrieve the pointer to the external transform (its address).
        if (const auto transformAddressFound = transformParameterMap.find("TransformAddress");
            transformAddressFound != transformParameterMap.end() && !transformAddressFound->second.empty() &&
            !transformAddressFound->second.front().empty() &&
            elx::Conversion::StringToValue(transformAddressFound->second.front(), externalTransform))
        {
          const auto transformFileName = "InitialTransform." + std::to_string(i) + '.' + outputFileNameExtension;

          // Write the external transform to file.
          const auto writer = itk::TransformFileWriter::New();
          writer->SetInput(externalTransform);
          writer->SetFileName(m_OutputDirectory + transformFileName);
          writer->Update();

          // Store the name of the written transform file.
          transformFound->second = { "File" };
          transformParameterMap["TransformFileName"] = { transformFileName };
          transformParameterMap.erase("TransformAddress");
        }
      }

      const auto transformParameterFileName = "InitialTransformParameters." + std::to_string(i) + ".txt";
      elx::ParameterObject::WriteParameterFile(transformParameterMap, m_OutputDirectory + transformParameterFileName);
      initialTransformParameterFileName = transformParameterFileName;
      ++i;
    }

    // Pass the last initial transform parameter file name to the argument map, in order to have it stored by
    // elx::TransformBase::ReadFromFile(), so that it can be retrieved later by
    // elx::TransformBase::GetInitialTransformParameterFileName(). Use "-tp", instead of "-t0", to avoid actual file
    // reading of the initial transforms, as they are already in memory.
    argumentMap["-tp"] = initialTransformParameterFileName;
  }

  const auto fixedImageDimensionString = std::to_string(FixedImageDimension);
  const auto fixedImagePixelTypeString = elx::PixelTypeToString<typename TFixedImage::PixelType>();
  const auto movingImageDimensionString = std::to_string(MovingImageDimension);

  const std::vector<TFixedImage *>  fixedInputImages = GetInputImages<TFixedImage>("FixedImage");
  const std::vector<TMovingImage *> movingInputImages = GetInputImages<TMovingImage>("MovingImage");

  // Cache the containers of internal images, to avoid casting the input images to the same internal pixel type more
  // than once, in the `for` loop below here.
  std::map<std::string, itk::SmartPointer<DataObjectContainerType>> fixedInternalImageContainers;
  std::map<std::string, itk::SmartPointer<DataObjectContainerType>> movingInternalImageContainers;

  // Run the (possibly multiple) registration(s)
  for (unsigned int i = 0; i < parameterMapVector.size(); ++i)
  {
    auto & parameterMap = parameterMapVector[i];

    // Lambda to create a FixedImageContainer or a MovingImageContainer.
    const auto createImageContainer =
      [](const auto & inputImages, const auto internalPixelTypeString, auto & imageContainers) {
        if (const auto found = imageContainers.find(internalPixelTypeString); found == imageContainers.end())
        {
          const auto imageContainer = DataObjectContainerType::New();

          for (const auto inputImage : inputImages)
          {
            const auto internalImage = CastToInternalPixelType<TFixedImage>(inputImage, internalPixelTypeString);
            imageContainer->push_back(internalImage);
          }
          imageContainers[internalPixelTypeString] = imageContainer;
          return imageContainer;
        }
        else
        {
          // There was an image container already for the specified internal pixel type. So just use that one!
          return found->second;
        }
      };

    // Set image dimension from input images (overrides user settings)
    SetParameterValueAndWarnOnOverride(parameterMap, "FixedImageDimension", fixedImageDimensionString);
    SetParameterValueAndWarnOnOverride(parameterMap, "MovingImageDimension", movingImageDimensionString);

    SetParameterValueAndWarnOnOverride(parameterMap, "ResultImagePixelType", fixedImagePixelTypeString);

    // Create new instance of ElastixMain
    const auto elastixMain = elx::ElastixMain::New();
    m_ElastixMain = elastixMain;

    // Set elastix levels
    elastixMain->SetElastixLevel(i);
    elastixMain->SetTotalNumberOfElastixLevels(parameterMapVector.size());

    // Set stuff we get from a previous registration
    elastixMain->SetInitialTransform(transform);
    elastixMain->SetFixedImageContainer(
      createImageContainer(fixedInputImages,
                           RetrievePixelTypeParameterValue(parameterMap, "FixedInternalImagePixelType"),
                           fixedInternalImageContainers));
    elastixMain->SetMovingImageContainer(
      createImageContainer(movingInputImages,
                           RetrievePixelTypeParameterValue(parameterMap, "MovingInternalImagePixelType"),
                           movingInternalImageContainers));
    elastixMain->SetFixedMaskContainer(fixedMaskContainer);
    elastixMain->SetMovingMaskContainer(movingMaskContainer);
    elastixMain->SetResultImageContainer(resultImageContainer);
    elastixMain->SetOriginalFixedImageDirectionFlat(fixedImageOriginalDirection);

    // Start registration
    unsigned int isError = 0;
    try
    {
      isError =
        ((i == 0) && !transformParameterMapVector.empty())
          ? elastixMain->RunWithInitialTransformParameterMaps(argumentMap, parameterMap, transformParameterMapVector)
          : elastixMain->Run(argumentMap, parameterMap);
    }
    catch (const itk::ExceptionObject & e)
    {
      itkExceptionMacro("Errors occurred during registration: " << e.what());
    }

    if (isError != 0)
    {
      itkExceptionMacro("Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn()).");
    }

    // Get stuff in order to put it in the next registration
    transform = elastixMain->GetFinalTransform();
    fixedMaskContainer = elastixMain->GetFixedMaskContainer();
    movingMaskContainer = elastixMain->GetMovingMaskContainer();
    resultImageContainer = elastixMain->GetResultImageContainer();
    fixedImageOriginalDirection = elastixMain->GetOriginalFixedImageDirectionFlat();

    transformParameterMapVector.push_back(elastixMain->GetTransformParameterMap());

    // TODO: Fix elastix corrupting default pixel value parameter
    transformParameterMapVector.back()["DefaultPixelValue"] = parameterMap["DefaultPixelValue"];
  } // End loop over registrations

  // Save result image
  if (resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 &&
      resultImageContainer->ElementAt(0).IsNotNull())
  {
    this->GraftOutput(resultImageContainer->ElementAt(0));
  }
  else
  {
    const auto & parameterMap = parameterMapVector.back();
    const auto   endOfParameterMap = parameterMap.cend();
    const bool   writeResultImage =
      std::find(parameterMap.cbegin(),
                endOfParameterMap,
                typename ParameterMapType::value_type{ "WriteResultImage", { "false" } }) == endOfParameterMap;

    if (writeResultImage)
    {
      itkExceptionMacro("Errors occurred during registration: Could not read result image.");
    }
  }

  // Save parameter map
  elx::ParameterObject::Pointer transformParameterObject = elx::ParameterObject::New();
  transformParameterObject->SetParameterMaps(transformParameterMapVector);
  this->SetNthOutput(1, transformParameterObject);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetParameterObject(ParameterObjectType * parameterObject)
{
  this->ProcessObject::SetInput("ParameterObject", parameterObject);
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetParameterObject() -> ParameterObjectType *
{
  return itkDynamicCastInDebugMode<ParameterObjectType *>(this->ProcessObject::GetInput("ParameterObject"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetParameterObject() const -> const ParameterObjectType *
{
  return itkDynamicCastInDebugMode<const ParameterObjectType *>(this->ProcessObject::GetInput("ParameterObject"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetTransformParameterObject() -> ParameterObjectType *
{
  return static_cast<ParameterObjectType *>(this->ProcessObject::GetOutput(1));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetTransformParameterObject() const -> const ParameterObjectType *
{
  return static_cast<const ParameterObjectType *>(this->ProcessObject::GetOutput(1));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetOutput() -> ResultImageType *
{
  return static_cast<ResultImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetOutput() const -> const ResultImageType *
{
  return static_cast<const ResultImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TFixedImage, typename TMovingImage>
DataObject *
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetOutput(unsigned int idx)
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TFixedImage, typename TMovingImage>
const DataObject *
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetOutput(unsigned int idx) const
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TFixedImage, typename TMovingImage>
ProcessObject::DataObjectPointer
ElastixRegistrationMethod<TFixedImage, TMovingImage>::MakeOutput(DataObjectPointerArraySizeType idx)
{
  if (idx == 1)
  {
    elx::ParameterObject::Pointer transformParameterObject = elx::ParameterObject::New();
    return transformParameterObject.GetPointer();
  }
  return Superclass::MakeOutput(idx);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetFixedImage(TFixedImage * fixedImage)
{
  this->RemoveInputsOfType("FixedImage");
  this->ProcessObject::SetInput("FixedImage", fixedImage);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::AddFixedImage(TFixedImage * fixedImage)
{
  if (this->ProcessObject::GetInput("FixedImage") == nullptr)
  {
    this->SetFixedImage(fixedImage);
  }
  else
  {
    this->ProcessObject::SetInput(this->MakeUniqueName("FixedImage"), fixedImage);
  }
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetFixedImage() const -> const FixedImageType *
{
  if (this->GetNumberOfInputsOfType("FixedImage") > 1)
  {
    itkExceptionMacro("Please provide an index when more than one fixed images are available.");
  }

  return itkDynamicCastInDebugMode<const TFixedImage *>(this->ProcessObject::GetInput("FixedImage"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetFixedImage(const unsigned int index) const
  -> const FixedImageType *
{
  unsigned int n = 0;
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType("FixedImage", inputName))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const TFixedImage *>(this->ProcessObject::GetInput(inputName));
      }

      ++n;
    }
  }

  itkExceptionMacro("Index exceeds the number of fixed images (index: " << index << ", number of fixed images: " << n
                                                                        << ")");
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfFixedImages() const
{
  return this->GetNumberOfInputsOfType("FixedImage");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetMovingImage(TMovingImage * movingImage)
{
  this->RemoveInputsOfType("MovingImage");
  this->ProcessObject::SetInput("MovingImage", movingImage);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::AddMovingImage(TMovingImage * movingImage)
{
  if (this->ProcessObject::GetInput("MovingImage") == nullptr)
  {
    this->SetMovingImage(movingImage);
  }
  else
  {
    this->ProcessObject::SetInput(this->MakeUniqueName("MovingImage"), movingImage);
  }
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetMovingImage() const -> const MovingImageType *
{
  if (this->GetNumberOfInputsOfType("MovingImage") > 1)
  {
    itkExceptionMacro("Please provide an index when more than one fixed images are available.");
  }

  return itkDynamicCastInDebugMode<const TMovingImage *>(this->ProcessObject::GetInput("MovingImage"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetMovingImage(const unsigned int index) const
  -> const MovingImageType *
{
  unsigned int n = 0;
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType("MovingImage", inputName))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const TMovingImage *>(this->ProcessObject::GetInput(inputName));
      }

      ++n;
    }
  }

  itkExceptionMacro("Index exceeds the number of moving images (index: " << index << ", number of moving images: " << n
                                                                         << ")");
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfMovingImages() const
{
  return this->GetNumberOfInputsOfType("MovingImage");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetFixedMask(FixedMaskType * fixedMask)
{
  this->RemoveInputsOfType("FixedMask");
  this->ProcessObject::SetInput("FixedMask", fixedMask);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::AddFixedMask(FixedMaskType * fixedMask)
{
  this->ProcessObject::SetInput(this->MakeUniqueName("FixedMask"), fixedMask);
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetFixedMask() const -> const FixedMaskType *
{
  return itkDynamicCastInDebugMode<const FixedMaskType *>(this->ProcessObject::GetInput("FixedMask"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetFixedMask(const unsigned int index) const
  -> const FixedMaskType *
{
  unsigned int n = 0;
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType("FixedMask", inputName))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const FixedMaskType *>(this->ProcessObject::GetInput(inputName));
      }

      ++n;
    }
  }

  itkExceptionMacro("Index exceeds the number of fixed masks (index: " << index << ", number of fixed masks: " << n
                                                                       << ")");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveFixedMask()
{
  this->RemoveInputsOfType("FixedMask");
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfFixedMasks() const
{
  return this->GetNumberOfInputsOfType("FixedMask");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetMovingMask(MovingMaskType * movingMask)
{
  this->RemoveInputsOfType("MovingMask");
  this->AddMovingMask(movingMask);
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::AddMovingMask(MovingMaskType * movingMask)
{
  this->ProcessObject::SetInput(this->MakeUniqueName("MovingMask"), movingMask);
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetMovingMask() const -> const MovingMaskType *
{
  return itkDynamicCastInDebugMode<const MovingMaskType *>(this->ProcessObject::GetInput("MovingMask"));
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetMovingMask(const unsigned int index) const
  -> const MovingMaskType *
{
  unsigned int n = 0;
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType("MovingMask", inputName))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const MovingMaskType *>(this->ProcessObject::GetInput(inputName));
      }

      ++n;
    }
  }

  itkExceptionMacro("Index exceeds the number of moving masks (index: " << index << ", number of moving masks: " << n
                                                                        << ")");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveMovingMask()
{
  this->RemoveInputsOfType("MovingMask");
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfMovingMasks() const
{
  return this->GetNumberOfInputsOfType("MovingMask");
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetInput(FixedImageType * fixedImage)
{
  this->SetFixedImage(fixedImage);
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetInput() const -> const FixedImageType *
{
  return this->GetFixedImage();
}


template <typename TFixedImage, typename TMovingImage>
const DataObject *
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetInput(DataObjectPointerArraySizeType index) const
{
  switch (index)
  {
    case 0:
      return this->GetFixedImage();
    case 1:
      return this->GetMovingImage();
    case 2:
      return this->GetParameterObject();
    default:
      return this->ProcessObject::GetInput(index);
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetInput(DataObjectPointerArraySizeType index, DataObject * input)
{
  switch (index)
  {
    case 0:
      this->SetFixedImage(itkDynamicCastInDebugMode<TFixedImage *>(input));
      break;
    case 1:
      this->SetMovingImage(itkDynamicCastInDebugMode<TMovingImage *>(input));
      break;
    case 2:
      this->SetParameterObject(itkDynamicCastInDebugMode<ParameterObjectType *>(input));
      break;
    default:
      this->ProcessObject::SetNthInput(index, input);
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetInitialTransformParameterFileName(std::string fileName)
{
  if (fileName.empty())
  {
    ResetInitialTransformAndModified();
  }
  else
  {
    if (m_InitialTransformParameterFileName != fileName)
    {
      ResetInitialTransformWithoutModified();
      m_InitialTransformParameterFileName = std::move(fileName);
      this->Modified();
    }
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetInitialTransformParameterObject(
  const elx::ParameterObject * const parameterObject)
{
  if (parameterObject)
  {
    if (m_InitialTransformParameterObject != parameterObject)
    {
      ResetInitialTransformWithoutModified();
      m_InitialTransformParameterObject = parameterObject;
      this->Modified();
    }
  }
  else
  {
    ResetInitialTransformAndModified();
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetInitialTransform(const TransformType * const transform)
{
  if (transform)
  {
    if (m_InitialTransform != transform)
    {
      ResetInitialTransformWithoutModified();
      m_InitialTransform = transform;
      this->Modified();
    }
  }
  else
  {
    ResetInitialTransformAndModified();
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetExternalInitialTransform(TransformType * const transform)
{
  if (transform)
  {
    if (m_ExternalInitialTransform != transform)
    {
      ResetInitialTransformWithoutModified();
      m_ExternalInitialTransform = transform;
      this->Modified();
    }
  }
  else
  {
    ResetInitialTransformAndModified();
  }
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetLogFileName(const std::string logFileName)
{
  m_LogFileName = logFileName;
  this->LogToFileOn();
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveLogFileName()
{
  m_LogFileName = "";
  this->LogToFileOff();
}


template <typename TFixedImage, typename TMovingImage>
std::string
ElastixRegistrationMethod<TFixedImage, TMovingImage>::MakeUniqueName(const DataObjectIdentifierType & inputName)
{
  return inputName + std::to_string(m_InputUID++);
}


template <typename TFixedImage, typename TMovingImage>
bool
ElastixRegistrationMethod<TFixedImage, TMovingImage>::IsInputOfType(const DataObjectIdentifierType & inputType,
                                                                    const DataObjectIdentifierType & inputName) const
{
  return std::strncmp(inputType.c_str(), inputName.c_str(), std::min(inputType.size(), inputName.size())) == 0;
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfInputsOfType(
  const DataObjectIdentifierType & inputType) const
{
  unsigned int n = 0;
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType(inputType, inputName))
    {
      ++n;
    }
  }

  return n;
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveInputsOfType(const DataObjectIdentifierType & inputType)
{
  for (const auto & inputName : this->GetInputNames())
  {
    if (this->IsInputOfType(inputType, inputName))
    {
      this->RemoveInput(inputName);
    }
  }
}


template <typename TFixedImage, typename TMovingImage>
unsigned int
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNumberOfTransforms() const
{
  const auto * const transformContainer = m_ElastixMain->GetElastixBase().GetTransformContainer();

  if ((transformContainer == nullptr) || transformContainer->empty())
  {
    return 0;
  }

  const auto * const elxTransformBase =
    dynamic_cast<ElastixTransformBaseType *>(transformContainer->front().GetPointer());

  return (elxTransformBase == nullptr) ? 0 : elxTransformBase->GetAsITKBaseType()->GetNumberOfTransforms();
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetNthTransform(const unsigned int n) const -> TransformType *
{
  const auto * const transformContainer = m_ElastixMain->GetElastixBase().GetTransformContainer();

  if ((transformContainer == nullptr) || transformContainer->empty())
  {
    return nullptr;
  }

  const auto * const elxTransformBase =
    dynamic_cast<ElastixTransformBaseType *>(transformContainer->front().GetPointer());

  if (elxTransformBase == nullptr)
  {
    return nullptr;
  }
  return elxTransformBase->GetAsITKBaseType()->GetNthTransform(n);
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GetCombinationTransform() const -> TransformType *
{
  const auto * const transformContainer = m_ElastixMain->GetElastixBase().GetTransformContainer();

  if ((transformContainer == nullptr) || transformContainer->empty())
  {
    return nullptr;
  }

  auto * const elxTransformBase = dynamic_cast<ElastixTransformBaseType *>(transformContainer->front().GetPointer());

  if (elxTransformBase == nullptr)
  {
    return nullptr;
  }
  return elxTransformBase->GetAsITKBaseType();
}


template <typename TFixedImage, typename TMovingImage>
auto
ElastixRegistrationMethod<TFixedImage, TMovingImage>::ConvertToItkTransform(const TransformType & elxTransform)
  -> SmartPointer<TransformType>
{
  const auto * const combinationTransform =
    dynamic_cast<const itk::AdvancedCombinationTransform<double, FixedImageDimension> *>(&elxTransform);

  const auto itkTransform = combinationTransform
                              ? elx::TransformIO::ConvertToCompositionOfItkTransforms(*combinationTransform)
                              : elx::TransformIO::ConvertToSingleItkTransform(elxTransform);
  if (itkTransform)
  {
    return itkTransform;
  }
  itkGenericExceptionMacro("Failed to convert transform object " << elxTransform);
}

} // namespace itk

#endif
