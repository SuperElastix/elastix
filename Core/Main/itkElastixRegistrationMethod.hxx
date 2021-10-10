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

#include "elxPixelType.h"
#include "itkElastixRegistrationMethod.h"

#include <algorithm> // For find.

namespace itk
{

template <typename TFixedImage, typename TMovingImage>
ElastixRegistrationMethod<TFixedImage, TMovingImage>::ElastixRegistrationMethod()
{
  this->SetPrimaryInputName("FixedImage");
  this->SetNumberOfIndexedOutputs(2);

  this->AddRequiredInputName("MovingImage", 1);
  this->AddRequiredInputName("ParameterObject", 2);

  this->m_InitialTransformParameterFileName = "";
  this->m_FixedPointSetFileName = "";
  this->m_MovingPointSetFileName = "";

  this->m_OutputDirectory = "";
  this->m_LogFileName = "";

  this->m_LogToConsole = false;
  this->m_LogToFile = false;

  this->m_NumberOfThreads = 0;

  ParameterObjectPointer defaultParameterObject = elastix::ParameterObject::New();
  defaultParameterObject->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("translation"));
  defaultParameterObject->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("affine"));
  defaultParameterObject->AddParameterMap(elastix::ParameterObject::GetDefaultParameterMap("bspline"));
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

  this->m_InputUID = 0;
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::GenerateData()
{
  // Force compiler to instantiate the image dimensions, otherwise we may get
  //   Undefined symbols for architecture x86_64:
  //     "elastix::ElastixRegistrationMethod<itk::Image<float, 2u> >::FixedImageDimension"
  // on some platforms.
  const unsigned int fixedImageDimension = FixedImageDimension;
  const unsigned int movingImageDimension = MovingImageDimension;

  DataObjectContainerPointer fixedImageContainer = DataObjectContainerType::New();
  DataObjectContainerPointer movingImageContainer = DataObjectContainerType::New();
  DataObjectContainerPointer fixedMaskContainer = nullptr;
  DataObjectContainerPointer movingMaskContainer = nullptr;
  DataObjectContainerPointer resultImageContainer = nullptr;
  ElastixMainObjectPointer   transform = nullptr;
  ParameterMapVectorType     transformParameterMapVector;
  FlatDirectionCosinesType   fixedImageOriginalDirection;

  // Split inputs into separate containers
  const NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType("FixedImage", inputNames[i]))
    {
      fixedImageContainer->push_back(this->ProcessObject::GetInput(inputNames[i]));
      continue;
    }

    if (this->IsInputOfType("MovingImage", inputNames[i]))
    {
      movingImageContainer->push_back(this->ProcessObject::GetInput(inputNames[i]));
      continue;
    }

    if (this->IsInputOfType("FixedMask", inputNames[i]))
    {
      if (fixedMaskContainer.IsNull())
      {
        fixedMaskContainer = DataObjectContainerType::New();
      }

      fixedMaskContainer->push_back(this->ProcessObject::GetInput(inputNames[i]));
      continue;
    }

    if (this->IsInputOfType("MovingMask", inputNames[i]))
    {
      if (movingMaskContainer.IsNull())
      {
        movingMaskContainer = DataObjectContainerType::New();
      }

      movingMaskContainer->push_back(this->ProcessObject::GetInput(inputNames[i]));
    }
  }

  // Set ParameterMap
  ParameterObjectPointer parameterObject =
    itkDynamicCastInDebugMode<elastix::ParameterObject *>(this->ProcessObject::GetInput("ParameterObject"));
  ParameterMapVectorType parameterMapVector = parameterObject->GetParameterMap();

  if (parameterMapVector.empty())
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  // Setup argument map
  ArgumentMapType argumentMap;

  if (!this->m_InitialTransformParameterFileName.empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-t0", this->m_InitialTransformParameterFileName));
  }

  if (!this->m_FixedPointSetFileName.empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-fp", this->m_FixedPointSetFileName));
  }

  if (!this->m_MovingPointSetFileName.empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-mp", this->m_MovingPointSetFileName));
  }

  // Setup output directory
  if (this->GetOutputDirectory().empty())
  {
    if (this->GetLogToFile())
    {
      itkExceptionMacro("LogToFileOn() requires an output directory to be specified.")
    }

    // There must be an "-out" as this is checked later in the code
    argumentMap.insert(ArgumentMapEntryType("-out", "output_path_not_set"));
  }
  else
  {
    if (!itksys::SystemTools::FileExists(this->GetOutputDirectory()))
    {
      itkExceptionMacro("Output directory \"" << this->GetOutputDirectory() << "\" does not exist.");
    }

    if (this->GetOutputDirectory().back() != '/' && this->GetOutputDirectory().back() != '\\')
    {
      this->SetOutputDirectory(this->GetOutputDirectory() + "/");
    }

    argumentMap.insert(ArgumentMapEntryType("-out", this->GetOutputDirectory()));
  }

  // Setup log file
  std::string logFileName;
  if (this->GetLogToFile())
  {
    if (this->GetLogFileName().empty())
    {
      logFileName = this->GetOutputDirectory() + "elastix.log";
    }
    else
    {
      logFileName = this->GetOutputDirectory() + this->GetLogFileName();
    }
  }

  // Set Number of threads
  if (this->m_NumberOfThreads > 0)
  {
    argumentMap.insert(ArgumentMapEntryType("-threads", std::to_string(this->m_NumberOfThreads)));
  }

  // Setup xout
  const elastix::xoutManager manager(logFileName, this->GetLogToFile(), this->GetLogToConsole());

  // Run the (possibly multiple) registration(s)
  for (unsigned int i = 0; i < parameterMapVector.size(); ++i)
  {
    // Set image dimension from input images (overrides user settings)
    parameterMapVector[i]["FixedImageDimension"] = ParameterValueVectorType(1, std::to_string(fixedImageDimension));
    parameterMapVector[i]["MovingImageDimension"] = ParameterValueVectorType(1, std::to_string(movingImageDimension));
    parameterMapVector[i]["ResultImagePixelType"] =
      ParameterValueVectorType(1, elastix::PixelType<typename TFixedImage::PixelType>::ToString());

    // Initial transform parameter files are handled via arguments and enclosing loop, not
    // InitialTransformParametersFileName
    if (parameterMapVector[i].find("InitialTransformParametersFileName") != parameterMapVector[i].end())
    {
      parameterMapVector[i]["InitialTransformParametersFileName"] = ParameterValueVectorType(1, "NoInitialTransform");
    }

    // Create new instance of ElastixMain
    ElastixMainPointer elastix = ElastixMainType::New();

    // Set elastix levels
    elastix->SetElastixLevel(i);
    elastix->SetTotalNumberOfElastixLevels(parameterMapVector.size());

    // Set stuff we get from a previous registration
    elastix->SetInitialTransform(transform);
    elastix->SetFixedImageContainer(fixedImageContainer);
    elastix->SetMovingImageContainer(movingImageContainer);
    elastix->SetFixedMaskContainer(fixedMaskContainer);
    elastix->SetMovingMaskContainer(movingMaskContainer);
    elastix->SetResultImageContainer(resultImageContainer);
    elastix->SetOriginalFixedImageDirectionFlat(fixedImageOriginalDirection);

    // Start registration
    unsigned int isError = 0;
    try
    {
      isError = elastix->Run(argumentMap, parameterMapVector[i]);
    }
    catch (itk::ExceptionObject & e)
    {
      itkExceptionMacro(<< "Errors occurred during registration: " << e.what());
    }

    if (isError != 0)
    {
      itkExceptionMacro(<< "Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn()).");
    }

    // Get stuff in order to put it in the next registration
    transform = elastix->GetFinalTransform();
    fixedImageContainer = elastix->GetFixedImageContainer();
    movingImageContainer = elastix->GetMovingImageContainer();
    fixedMaskContainer = elastix->GetFixedMaskContainer();
    movingMaskContainer = elastix->GetMovingMaskContainer();
    resultImageContainer = elastix->GetResultImageContainer();
    fixedImageOriginalDirection = elastix->GetOriginalFixedImageDirectionFlat();

    transformParameterMapVector.push_back(elastix->GetTransformParametersMap());
    if (i > 0)
    {
      transformParameterMapVector[i]["InitialTransformParametersFileName"] =
        ParameterValueVectorType(1, std::to_string(i - 1));
    }

    // TODO: Fix elastix corrupting default pixel value parameter
    transformParameterMapVector.back()["DefaultPixelValue"] = parameterMapVector[i]["DefaultPixelValue"];
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
      itkExceptionMacro("Errors occured during registration: Could not read result image.");
    }
  }

  // Save parameter map
  elastix::ParameterObject::Pointer transformParameterObject = elastix::ParameterObject::New();
  transformParameterObject->SetParameterMap(transformParameterMapVector);
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
    elastix::ParameterObject::Pointer transformParameterObject = elastix::ParameterObject::New();
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
  if (this->ProcessObject::GetInput("FixedImage") == ITK_NULLPTR)
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
  unsigned int  n = 0;
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType("FixedImage", inputNames[i]))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const TFixedImage *>(this->ProcessObject::GetInput(inputNames[i]));
      }

      n++;
    }
  }

  itkExceptionMacro(<< "Index exceeds the number of fixed images (index: " << index << ", number of fixed images: " << n
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
  if (this->ProcessObject::GetInput("MovingImage") == ITK_NULLPTR)
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
  unsigned int  n = 0;
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType("MovingImage", inputNames[i]))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const TMovingImage *>(this->ProcessObject::GetInput(inputNames[i]));
      }

      n++;
    }
  }

  itkExceptionMacro(<< "Index exceeds the number of moving images (index: " << index
                    << ", number of moving images: " << n << ")");
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
  unsigned int  n = 0;
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType("FixedMask", inputNames[i]))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const FixedMaskType *>(this->ProcessObject::GetInput(inputNames[i]));
      }

      n++;
    }
  }

  itkExceptionMacro(<< "Index exceeds the number of fixed masks (index: " << index << ", number of fixed masks: " << n
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
  unsigned int  n = 0;
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType("MovingMask", inputNames[i]))
    {
      if (index == n)
      {
        return itkDynamicCastInDebugMode<const MovingMaskType *>(this->ProcessObject::GetInput(inputNames[i]));
      }

      n++;
    }
  }

  itkExceptionMacro(<< "Index exceeds the number of moving masks (index: " << index << ", number of moving masks: " << n
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
ElastixRegistrationMethod<TFixedImage, TMovingImage>::SetLogFileName(const std::string logFileName)
{
  this->m_LogFileName = logFileName;
  this->LogToFileOn();
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveLogFileName()
{
  this->m_LogFileName = "";
  this->LogToFileOff();
}


template <typename TFixedImage, typename TMovingImage>
std::string
ElastixRegistrationMethod<TFixedImage, TMovingImage>::MakeUniqueName(const DataObjectIdentifierType & inputName)
{
  return inputName + std::to_string(this->m_InputUID++);
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
  unsigned int  n = 0;
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType(inputType, inputNames[i]))
    {
      n++;
    }
  }

  return n;
}


template <typename TFixedImage, typename TMovingImage>
void
ElastixRegistrationMethod<TFixedImage, TMovingImage>::RemoveInputsOfType(const DataObjectIdentifierType & inputType)
{
  NameArrayType inputNames = this->GetInputNames();
  for (unsigned int i = 0; i < inputNames.size(); ++i)
  {
    if (this->IsInputOfType(inputType, inputNames[i]))
    {
      this->RemoveInput(inputNames[i]);
    }
  }
}


} // namespace itk

#endif
