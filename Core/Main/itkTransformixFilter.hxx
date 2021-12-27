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
#ifndef itkTransformixFilter_hxx
#define itkTransformixFilter_hxx

#include "itkTransformixFilter.h"
#include "elxPixelType.h"

namespace itk
{

template <typename TMovingImage>
TransformixFilter<TMovingImage>::TransformixFilter()
{
  this->SetPrimaryInputName("MovingImage");
  this->AddRequiredInputName("TransformParameterObject", 1);

  this->SetOutput("ResultDeformationField", this->MakeOutput("ResultDeformationField"));

  this->m_FixedPointSetFileName = "";
  this->m_ComputeSpatialJacobian = false;
  this->m_ComputeDeterminantOfSpatialJacobian = false;
  this->m_ComputeDeformationField = false;

  this->m_OutputDirectory = "";
  this->m_LogFileName = "";

  this->m_LogToConsole = false;
  this->m_LogToFile = false;
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::GenerateData()
{
  // Force compiler to instantiate the image dimension, otherwise we may get
  //   Undefined symbols for architecture x86_64:
  //     "elastix::TransformixFilter<itk::Image<float, 2u> >::MovingImageDimension"
  // on some platforms.
  const unsigned int movingImageDimension = MovingImageDimension;

  if (this->IsEmpty(this->GetMovingImage()) && this->GetFixedPointSetFileName().empty() &&
      !this->GetComputeSpatialJacobian() && !this->GetComputeDeterminantOfSpatialJacobian() &&
      !this->GetComputeDeformationField())
  {
    itkExceptionMacro(
      "Expected at least one of SetMovingImage(), SetFixedPointSetFileName() ComputeSpatialJacobianOn(), "
      "ComputeDeterminantOfSpatialJacobianOn() or ComputeDeformationFieldOn(), to be active.\"");
  }

  // TODO: Patch upstream transformix to split this into seperate arguments
  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  if (this->GetComputeDeformationField() && !this->GetFixedPointSetFileName().empty())
  {
    itkExceptionMacro(<< "For backwards compatibility, only one of ComputeDeformationFieldOn() or "
                         "SetFixedPointSetFileName() can be active at any one time.")
  }

  // Setup argument map which transformix uses internally ito figure out what needs to be done
  ArgumentMapType argumentMap;

  if (this->GetComputeSpatialJacobian())
  {
    argumentMap.insert(ArgumentMapEntryType("-jacmat", "all"));
  }

  if (this->GetComputeDeterminantOfSpatialJacobian())
  {
    argumentMap.insert(ArgumentMapEntryType("-jac", "all"));
  }

  if (this->GetComputeDeformationField())
  {
    argumentMap.insert(ArgumentMapEntryType("-def", "all"));
  }

  if (!this->GetFixedPointSetFileName().empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-def", this->GetFixedPointSetFileName()));
  }

  // Setup output directory
  // Only the input "MovingImage" does not require an output directory
  if ((this->GetComputeSpatialJacobian() || this->GetComputeDeterminantOfSpatialJacobian() ||
       this->GetComputeDeformationField() || !this->GetFixedPointSetFileName().empty() || this->GetLogToFile()) &&
      this->GetOutputDirectory().empty())
  {
    this->SetOutputDirectory(".");
  }

  if (!this->GetOutputDirectory().empty() && !itksys::SystemTools::FileExists(this->GetOutputDirectory()))
  {
    itkExceptionMacro("Output directory \"" << this->GetOutputDirectory() << "\" does not exist.")
  }

  if (this->GetOutputDirectory().empty())
  {
    // There must be an "-out", this is checked later in the code
    argumentMap.insert(ArgumentMapEntryType("-out", "output_path_not_set"));
  }
  else
  {
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
      logFileName = this->GetOutputDirectory() + "transformix.log";
    }
    else
    {
      logFileName = this->GetOutputDirectory() + this->GetLogFileName();
    }
  }

  // Setup xout
  const elx::xoutManager manager(logFileName, this->GetLogToFile(), this->GetLogToConsole());

  // Instantiate transformix
  TransformixMainPointer transformix = TransformixMainType::New();

  // Setup transformix for warping input image if given
  DataObjectContainerPointer inputImageContainer = nullptr;
  if (!this->IsEmpty(this->GetMovingImage()))
  {
    inputImageContainer = DataObjectContainerType::New();
    inputImageContainer->InsertElement(0, const_cast<InputImageType *>(this->GetMovingImage()));
    transformix->SetInputImageContainer(inputImageContainer);
  }

  // Get ParameterMap
  ParameterObjectPointer transformParameterObject = this->GetTransformParameterObject();
  ParameterMapVectorType transformParameterMapVector = transformParameterObject->GetParameterMap();

  // Assert user did not set empty parameter map
  if (transformParameterMapVector.empty())
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  // Set pixel types from input image, override user settings
  for (unsigned int i = 0; i < transformParameterMapVector.size(); ++i)
  {
    transformParameterMapVector[i]["FixedImageDimension"] =
      ParameterValueVectorType(1, std::to_string(movingImageDimension));
    transformParameterMapVector[i]["MovingImageDimension"] =
      ParameterValueVectorType(1, std::to_string(movingImageDimension));
    transformParameterMapVector[i]["ResultImagePixelType"] =
      ParameterValueVectorType(1, elastix::PixelType<typename TMovingImage::PixelType>::ToString());

    if (i > 0)
    {
      transformParameterMapVector[i]["InitialTransformParametersFileName"] =
        ParameterValueVectorType(1, std::to_string(i - 1));
    }
  }

  // Run transformix
  unsigned int isError = 0;
  try
  {
    isError = transformix->Run(argumentMap, transformParameterMapVector);
  }
  catch (itk::ExceptionObject & e)
  {
    itkExceptionMacro("Errors occured during execution: " << e.what());
  }

  if (isError != 0)
  {
    itkExceptionMacro("Internal transformix error: See transformix log (use LogToConsoleOn() or LogToFileOn())");
  }

  // Save result image
  DataObjectContainerPointer resultImageContainer = transformix->GetResultImageContainer();
  if (resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 &&
      resultImageContainer->ElementAt(0).IsNotNull())
  {
    this->GraftOutput(resultImageContainer->ElementAt(0));
  }
  // Optionally, save result deformation field
  DataObjectContainerPointer resultDeformationFieldContainer = transformix->GetResultDeformationFieldContainer();
  if (resultDeformationFieldContainer.IsNotNull() && resultDeformationFieldContainer->Size() > 0 &&
      resultDeformationFieldContainer->ElementAt(0).IsNotNull())
  {
    this->GraftOutput("ResultDeformationField", resultDeformationFieldContainer->ElementAt(0));
  }
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::MakeOutput(const DataObjectIdentifierType & key) -> DataObjectPointer
{
  if (key == "ResultDeformationField")
  {
    return OutputDeformationFieldType::New().GetPointer();
  }
  else
  {
    // Primary and all other outputs default to ResultImage.
    return TMovingImage::New().GetPointer();
  }
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::GenerateOutputInformation()
{

  // Get pointers to the input and output
  const ParameterObjectType * transformParameterObjectPtr = this->GetTransformParameterObject();

  if (transformParameterObjectPtr->GetNumberOfParameterMaps() == 0)
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  OutputImageType *            outputPtr = this->GetOutput();
  OutputDeformationFieldType * outputOutputDeformationFieldPtr = this->GetOutputDeformationField();

  itkAssertInDebugAndIgnoreInReleaseMacro(transformParameterObjectPtr != ITK_NULLPTR);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputPtr != ITK_NULLPTR);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputOutputDeformationFieldPtr != ITK_NULLPTR);

  // Get world coordinate system from the last map
  const unsigned int     lastIndex = transformParameterObjectPtr->GetNumberOfParameterMaps() - 1;
  const ParameterMapType transformParameterMap = transformParameterObjectPtr->GetParameterMap(lastIndex);

  ParameterMapType::const_iterator spacingMapIter = transformParameterMap.find("Spacing");
  if (spacingMapIter == transformParameterMap.end())
  {
    itkExceptionMacro("No entry Spacing found in transformParameterMap");
  }
  const ParameterValueVectorType spacingStrings = spacingMapIter->second;

  ParameterMapType::const_iterator sizeMapIter = transformParameterMap.find("Size");
  if (sizeMapIter == transformParameterMap.end())
  {
    itkExceptionMacro("No entry Size found in transformParameterMap");
  }
  const ParameterValueVectorType sizeStrings = sizeMapIter->second;

  ParameterMapType::const_iterator indexMapIter = transformParameterMap.find("Index");
  if (indexMapIter == transformParameterMap.end())
  {
    itkExceptionMacro("No entry Index found in transformParameterMap");
  }
  const ParameterValueVectorType indexStrings = indexMapIter->second;

  ParameterMapType::const_iterator originMapIter = transformParameterMap.find("Origin");
  if (originMapIter == transformParameterMap.end())
  {
    itkExceptionMacro("No entry Origin found in transformParameterMap");
  }
  const ParameterValueVectorType originStrings = originMapIter->second;

  ParameterMapType::const_iterator directionMapIter = transformParameterMap.find("Direction");
  if (directionMapIter == transformParameterMap.end())
  {
    itkExceptionMacro("No entry Direction found in transformParameterMap");
  }
  const ParameterValueVectorType directionStrings = directionMapIter->second;

  typename TMovingImage::SpacingType   outputSpacing;
  typename TMovingImage::SizeType      outputSize;
  typename TMovingImage::IndexType     outputStartIndex;
  typename TMovingImage::PointType     outputOrigin;
  typename TMovingImage::DirectionType outputDirection;

  for (unsigned int i = 0; i < TMovingImage::ImageDimension; ++i)
  {
    outputSpacing[i] = std::atof(spacingStrings[i].c_str());
    outputSize[i] = std::atoi(sizeStrings[i].c_str());
    outputStartIndex[i] = std::atoi(indexStrings[i].c_str());
    outputOrigin[i] = std::atof(originStrings[i].c_str());
    for (unsigned int j = 0; j < TMovingImage::ImageDimension; ++j)
    {
      outputDirection(j, i) = std::atof(directionStrings[i * TMovingImage::ImageDimension + j].c_str());
    }
  }

  outputPtr->SetSpacing(outputSpacing);
  outputOutputDeformationFieldPtr->SetSpacing(outputSpacing);
  outputPtr->SetOrigin(outputOrigin);
  outputOutputDeformationFieldPtr->SetOrigin(outputOrigin);
  outputPtr->SetDirection(outputDirection);
  outputOutputDeformationFieldPtr->SetDirection(outputDirection);

  // Set region
  typename TMovingImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize(outputSize);
  outputLargestPossibleRegion.SetIndex(outputStartIndex);

  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
  outputOutputDeformationFieldPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);

  outputPtr->SetNumberOfComponentsPerPixel(1);
  outputOutputDeformationFieldPtr->SetNumberOfComponentsPerPixel(TMovingImage::ImageDimension);
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::SetMovingImage(TMovingImage * inputImage)
{
  this->ProcessObject::SetInput("MovingImage", inputImage);
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetMovingImage() const -> const InputImageType *
{
  return itkDynamicCastInDebugMode<const TMovingImage *>(this->ProcessObject::GetInput("MovingImage"));
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::RemoveMovingImage()
{
  this->ProcessObject::RemoveInput("MovingImage");
}

template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::SetInput(InputImageType * inputImage)
{
  this->ProcessObject::SetInput("MovingImage", inputImage);
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetInput() const -> const InputImageType *
{
  return itkDynamicCastInDebugMode<const TMovingImage *>(this->ProcessObject::GetInput("MovingImage"));
}

template <typename TMovingImage>
const DataObject *
TransformixFilter<TMovingImage>::GetInput(DataObjectPointerArraySizeType index) const
{
  switch (index)
  {
    case 0:
      return this->GetMovingImage();
    case 1:
      return this->GetTransformParameterObject();
    default:
      return this->ProcessObject::GetInput(index);
  }
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::SetInput(DataObjectPointerArraySizeType index, DataObject * input)
{
  switch (index)
  {
    case 0:
      this->SetMovingImage(itkDynamicCastInDebugMode<TMovingImage *>(input));
      break;
    case 1:
      this->SetTransformParameterObject(itkDynamicCastInDebugMode<ParameterObjectType *>(input));
      break;
    default:
      this->ProcessObject::SetNthInput(index, input);
  }
}

template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::SetTransformParameterObject(ParameterObjectType * parameterObject)
{
  this->ProcessObject::SetInput("TransformParameterObject", parameterObject);
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetTransformParameterObject() -> ParameterObjectType *
{
  return itkDynamicCastInDebugMode<ParameterObjectType *>(this->ProcessObject::GetInput("TransformParameterObject"));
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetTransformParameterObject() const -> const ParameterObjectType *
{
  return itkDynamicCastInDebugMode<const ParameterObjectType *>(
    this->ProcessObject::GetInput("TransformParameterObject"));
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetOutputDeformationField() -> OutputDeformationFieldType *
{
  return itkDynamicCastInDebugMode<OutputDeformationFieldType *>(
    this->itk::ProcessObject::GetOutput("ResultDeformationField"));
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetOutputDeformationField() const -> const OutputDeformationFieldType *
{
  return itkDynamicCastInDebugMode<const OutputDeformationFieldType *>(
    this->itk::ProcessObject::GetOutput("ResultDeformationField"));
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetOutput() -> OutputImageType *
{
  return static_cast<OutputImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TMovingImage>
auto
TransformixFilter<TMovingImage>::GetOutput() const -> const OutputImageType *
{
  return static_cast<const OutputImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TMovingImage>
DataObject *
TransformixFilter<TMovingImage>::GetOutput(unsigned int idx)
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TMovingImage>
const DataObject *
TransformixFilter<TMovingImage>::GetOutput(unsigned int idx) const
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TMovingImage>
bool
TransformixFilter<TMovingImage>::IsEmpty(const InputImageType * inputImage)
{
  if (!inputImage)
  {
    return true;
  }

  typename TMovingImage::RegionType region = inputImage->GetLargestPossibleRegion();
  return region.GetNumberOfPixels() == 0;
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::SetLogFileName(std::string logFileName)
{
  this->m_LogFileName = logFileName;
  this->LogToFileOn();
}


template <typename TMovingImage>
void
TransformixFilter<TMovingImage>::RemoveLogFileName()
{
  this->m_LogFileName = "";
  this->LogToFileOff();
}


} // namespace itk

#endif
