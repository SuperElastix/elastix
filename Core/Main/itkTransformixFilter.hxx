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

#include "elxLibUtilities.h"
#include "elxPixelTypeToString.h"
#include "elxTransformBase.h"
#include "elxTransformIO.h"
#include "elxDefaultConstruct.h"

#include <itkCompositeTransform.h>

#include <cassert>
#include <memory> // For unique_ptr.

namespace itk
{

template <typename TImage>
TransformixFilter<TImage>::TransformixFilter()
{
  this->SetPrimaryInputName("MovingImage");
  this->AddRequiredInputName("TransformParameterObject", 1);

  this->SetOutput("ResultDeformationField", this->MakeOutput("ResultDeformationField"));
}


template <typename TImage>
void
TransformixFilter<TImage>::GenerateData()
{
  using elx::LibUtilities::CastToInternalPixelType;
  using elx::LibUtilities::RetrievePixelTypeParameterValue;
  using elx::LibUtilities::SetParameterValueAndWarnOnOverride;

  if (this->IsEmpty(this->GetMovingImage()) && m_FixedPointSetFileName.empty() && !m_ComputeSpatialJacobian &&
      !m_ComputeDeterminantOfSpatialJacobian && !m_ComputeDeformationField)
  {
    itkExceptionMacro(
      "Expected at least one of SetMovingImage(), SetFixedPointSetFileName() ComputeSpatialJacobianOn(), "
      "ComputeDeterminantOfSpatialJacobianOn() or ComputeDeformationFieldOn(), to be active.\"");
  }

  // TODO: Patch upstream transformix to split this into seperate arguments
  // Transformix uses "-def" for path to point sets AND as flag for writing deformation field
  if (m_ComputeDeformationField && !m_FixedPointSetFileName.empty())
  {
    itkExceptionMacro("For backwards compatibility, only one of ComputeDeformationFieldOn() or "
                      "SetFixedPointSetFileName() can be active at any one time.");
  }

  // Setup argument map which transformix uses internally ito figure out what needs to be done
  ArgumentMapType argumentMap;

  if (m_ComputeSpatialJacobian)
  {
    argumentMap.insert(ArgumentMapEntryType("-jacmat", "all"));
  }

  if (m_ComputeDeterminantOfSpatialJacobian)
  {
    argumentMap.insert(ArgumentMapEntryType("-jac", "all"));
  }

  if (m_ComputeDeformationField)
  {
    argumentMap.insert(ArgumentMapEntryType("-def", "all"));
  }

  if (!m_FixedPointSetFileName.empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-def", m_FixedPointSetFileName));
  }

  if (!m_TransformParameterFileName.empty())
  {
    argumentMap.insert(ArgumentMapEntryType("-tp", m_TransformParameterFileName));
  }

  // Setup output directory
  // Only the input "MovingImage" does not require an output directory
  if ((m_ComputeSpatialJacobian || m_ComputeDeterminantOfSpatialJacobian || m_ComputeDeformationField ||
       !m_FixedPointSetFileName.empty() || m_LogToFile) &&
      m_OutputDirectory.empty())
  {
    this->SetOutputDirectory(".");
  }

  if (!m_OutputDirectory.empty() && !itksys::SystemTools::FileExists(m_OutputDirectory))
  {
    itkExceptionMacro("Output directory \"" << m_OutputDirectory << "\" does not exist.");
  }

  if (!m_OutputDirectory.empty())
  {
    if (m_OutputDirectory.back() != '/' && m_OutputDirectory.back() != '\\')
    {
      this->SetOutputDirectory(m_OutputDirectory + "/");
    }

    argumentMap.insert(ArgumentMapEntryType("-out", m_OutputDirectory));
  }

  // Setup log file
  const std::string logFileName = m_OutputDirectory + (m_LogFileName.empty() ? "transformix.log" : m_LogFileName);

  // Setup logging.
  const elx::log::guard logGuard(logFileName,
                                 m_EnableOutput && m_LogToFile,
                                 m_EnableOutput && m_LogToConsole,
                                 static_cast<elastix::log::level>(m_LogLevel));

  // Instantiate transformix
  const auto transformixMain = elx::TransformixMain::New();
  m_TransformixMain = transformixMain;

  // Get ParameterMap
  ParameterObjectPointer transformParameterObject = this->GetTransformParameterObject();
  ParameterMapVectorType transformParameterMapVector = transformParameterObject->GetParameterMaps();

  // Assert user did not set empty parameter map
  if (transformParameterMapVector.empty())
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  // Setup transformix for warping input image if given
  DataObjectContainerPointer inputImageContainer = nullptr;

  if (const auto movingImage = itkDynamicCastInDebugMode<TImage *>(this->ProcessObject::GetInput("MovingImage"));
      !Self::IsEmpty(movingImage))
  {
    // Note that the internal pixel type is only retrieved from the very first transform parameter map.
    const auto internalPixelTypeString =
      RetrievePixelTypeParameterValue(transformParameterMapVector.front(), "MovingInternalImagePixelType");
    const auto internalImage = CastToInternalPixelType<TImage>(movingImage, internalPixelTypeString);

    inputImageContainer = DataObjectContainerType::New();
    inputImageContainer->push_back(internalImage);
    transformixMain->SetInputImageContainer(inputImageContainer);
  }

  if (m_Transform)
  {
    // Adjust the local transformParameterMap according to this m_Transform.

    const auto transformToMap = [](const itk::TransformBase & transform, auto & transformParameterMap) {
      transformParameterMap["ITKTransformFixedParameters"] =
        elx::Conversion::ToVectorOfStrings(transform.GetFixedParameters());
      transformParameterMap["ITKTransformParameters"] = elx::Conversion::ToVectorOfStrings(transform.GetParameters());
      SetParameterValueAndWarnOnOverride(
        transformParameterMap, "ITKTransformType", transform.GetTransformTypeAsString());
      SetParameterValueAndWarnOnOverride(
        transformParameterMap,
        "Transform",
        elx::TransformIO::ConvertITKNameOfClassToElastixClassName(transform.GetNameOfClass()));
    };
    const auto compositeTransform =
      dynamic_cast<const CompositeTransform<double, MovingImageDimension> *>(&*m_Transform);

    if (compositeTransform)
    {
      const auto & transformQueue = compositeTransform->GetTransformQueue();

      const auto numberOfTransforms = transformQueue.size();

      if (numberOfTransforms == 0)
      {
        itkExceptionMacro(
          "The specified composite transform has no subtransforms! At least one subtransform is required!");
      }

      if (numberOfTransforms != transformParameterMapVector.size())
      {
        // The last TransformParameterMap is special, as it needs to be used for the final transformation.
        auto lastTransformParameterMap = transformParameterMapVector.back();
        transformParameterMapVector.resize(numberOfTransforms);
        transformParameterMapVector.back() = std::move(lastTransformParameterMap);
      }
      for (unsigned int i = 0; i < numberOfTransforms; ++i)
      {
        auto &     transformParameterMap = transformParameterMapVector[numberOfTransforms - i - 1];
        const auto transform = transformQueue[i];

        if (transform == nullptr)
        {
          itkExceptionMacro("One of the subtransforms of the specified composite transform is null!");
        }
        transformToMap(*transform, transformParameterMap);
      }
    }
    else
    {
      // Assume in this case that it is just a single transform.
      assert((dynamic_cast<const MultiTransform<double, MovingImageDimension> *>(&*m_Transform)) == nullptr);

      // For a single transform, there should be only a single transform parameter map.
      auto transformParameterMap = std::move(transformParameterMapVector.back());
      transformToMap(*m_Transform, transformParameterMap);
      transformParameterMapVector.clear();
      transformParameterMapVector.push_back(std::move(transformParameterMap));
    }
  }

  if (m_ExternalTransform)
  {
    // External transforms should use "ResampleInterpolator" and the output image domain specification
    // (Size/Spacing/Origin/Index/Direction) of the last transform parameter map.
    auto transformParameterMap = std::move(transformParameterMapVector.back());
    SetParameterValueAndWarnOnOverride(transformParameterMap, "Transform", "ExternalTransform");
    SetParameterValueAndWarnOnOverride(
      transformParameterMap, "TransformAddress", elx::Conversion::ObjectPtrToString(m_ExternalTransform));
    transformParameterMapVector.clear();
    transformParameterMapVector.push_back(std::move(transformParameterMap));
  }

  const auto movingImageDimensionString = std::to_string(MovingImageDimension);
  const auto movingImagePixelTypeString = elx::PixelTypeToString<typename TImage::PixelType>();

  // Set pixel types from input image, override user settings
  for (auto & transformParameterMap : transformParameterMapVector)
  {
    SetParameterValueAndWarnOnOverride(transformParameterMap, "FixedImageDimension", movingImageDimensionString);
    SetParameterValueAndWarnOnOverride(transformParameterMap, "MovingImageDimension", movingImageDimensionString);
    SetParameterValueAndWarnOnOverride(transformParameterMap, "ResultImagePixelType", movingImagePixelTypeString);
  }

  // Run transformix
  unsigned int isError = 0;
  try
  {
    isError = transformixMain->Run(argumentMap, transformParameterMapVector, m_CombinationTransform);

    if (m_InputMesh)
    {
      m_OutputMesh = nullptr;

      const auto * const transformContainer = transformixMain->GetElastixBase().GetTransformContainer();

      if ((transformContainer != nullptr) && (!transformContainer->empty()))
      {
        const auto transformBase = dynamic_cast<elx::TransformBase<elx::ElastixTemplate<TImage, TImage>> *>(
          transformContainer->front().GetPointer());

        if (transformBase)
        {
          m_OutputMesh = transformBase->TransformMesh(*m_InputMesh);
        }
      }
    }
  }
  catch (const itk::ExceptionObject & e)
  {
    itkExceptionMacro("Errors occurred during execution: " << e.what());
  }

  if (isError != 0)
  {
    itkExceptionMacro("Internal transformix error: See transformix log (use LogToConsoleOn() or LogToFileOn())");
  }

  // Save result image
  DataObjectContainerPointer resultImageContainer = transformixMain->GetResultImageContainer();
  if (resultImageContainer.IsNotNull() && resultImageContainer->Size() > 0 &&
      resultImageContainer->ElementAt(0).IsNotNull())
  {
    this->GraftOutput(resultImageContainer->ElementAt(0));
  }
  // Optionally, save result deformation field
  DataObjectContainerPointer resultDeformationFieldContainer = transformixMain->GetResultDeformationFieldContainer();
  if (resultDeformationFieldContainer.IsNotNull() && resultDeformationFieldContainer->Size() > 0 &&
      resultDeformationFieldContainer->ElementAt(0).IsNotNull())
  {
    this->GraftOutput("ResultDeformationField", resultDeformationFieldContainer->ElementAt(0));
  }
}


template <typename TImage>
auto
TransformixFilter<TImage>::ComputeSpatialJacobianDeterminantImage() const
  -> SmartPointer<SpatialJacobianDeterminantImageType>
{
  const auto transformBase = GetFirstElastixTransformBase();
  return transformBase ? transformBase->ComputeSpatialJacobianDeterminantImage() : nullptr;
}


template <typename TImage>
auto
TransformixFilter<TImage>::ComputeSpatialJacobianMatrixImage() const -> SmartPointer<SpatialJacobianMatrixImageType>
{
  const auto transformBase = GetFirstElastixTransformBase();
  return transformBase ? transformBase->ComputeSpatialJacobianMatrixImage() : nullptr;
}


template <typename TImage>
auto
TransformixFilter<TImage>::MakeOutput(const DataObjectIdentifierType & key) -> DataObjectPointer
{
  if (key == "ResultDeformationField")
  {
    return OutputDeformationFieldType::New().GetPointer();
  }
  else
  {
    // Primary and all other outputs default to ResultImage.
    return TImage::New().GetPointer();
  }
}


template <typename TImage>
void
TransformixFilter<TImage>::GenerateOutputInformation()
{

  // Get pointers to the input and output
  const ParameterObjectType * transformParameterObjectPtr = this->GetTransformParameterObject();

  if (transformParameterObjectPtr->GetNumberOfParameterMaps() == 0)
  {
    itkExceptionMacro("Empty parameter map in parameter object.");
  }

  OutputImageType *            outputPtr = this->GetOutput();
  OutputDeformationFieldType * outputOutputDeformationFieldPtr = this->GetOutputDeformationField();

  itkAssertInDebugAndIgnoreInReleaseMacro(transformParameterObjectPtr != nullptr);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputPtr != nullptr);
  itkAssertInDebugAndIgnoreInReleaseMacro(outputOutputDeformationFieldPtr != nullptr);

  // Get world coordinate system from the last map
  const unsigned int     lastIndex = transformParameterObjectPtr->GetNumberOfParameterMaps() - 1;
  const ParameterMapType transformParameterMap = transformParameterObjectPtr->GetParameterMap(lastIndex);

  const auto getTransformParameter = [&transformParameterMap](const char * const parameterName) {
    const auto it = transformParameterMap.find(parameterName);
    if (it == transformParameterMap.end())
    {
      itkGenericExceptionMacro("No entry " << parameterName << " found in transformParameterMap");
    }
    return it->second;
  };

  const ParameterValueVectorType spacingStrings = getTransformParameter("Spacing");
  const ParameterValueVectorType sizeStrings = getTransformParameter("Size");
  const ParameterValueVectorType indexStrings = getTransformParameter("Index");
  const ParameterValueVectorType originStrings = getTransformParameter("Origin");
  const ParameterValueVectorType directionStrings = getTransformParameter("Direction");

  typename TImage::SpacingType   outputSpacing;
  typename TImage::SizeType      outputSize;
  typename TImage::IndexType     outputStartIndex;
  typename TImage::PointType     outputOrigin;
  typename TImage::DirectionType outputDirection;

  for (unsigned int i = 0; i < TImage::ImageDimension; ++i)
  {
    outputSpacing[i] = std::atof(spacingStrings[i].c_str());
    outputSize[i] = std::atoi(sizeStrings[i].c_str());
    outputStartIndex[i] = std::atoi(indexStrings[i].c_str());
    outputOrigin[i] = std::atof(originStrings[i].c_str());
    for (unsigned int j = 0; j < TImage::ImageDimension; ++j)
    {
      outputDirection(j, i) = std::atof(directionStrings[i * TImage::ImageDimension + j].c_str());
    }
  }

  outputPtr->SetSpacing(outputSpacing);
  outputOutputDeformationFieldPtr->SetSpacing(outputSpacing);
  outputPtr->SetOrigin(outputOrigin);
  outputOutputDeformationFieldPtr->SetOrigin(outputOrigin);
  outputPtr->SetDirection(outputDirection);
  outputOutputDeformationFieldPtr->SetDirection(outputDirection);

  // Set region
  typename TImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion.SetSize(outputSize);
  outputLargestPossibleRegion.SetIndex(outputStartIndex);

  outputPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);
  outputOutputDeformationFieldPtr->SetLargestPossibleRegion(outputLargestPossibleRegion);

  outputPtr->SetNumberOfComponentsPerPixel(1);
  outputOutputDeformationFieldPtr->SetNumberOfComponentsPerPixel(TImage::ImageDimension);
}


template <typename TImage>
void
TransformixFilter<TImage>::SetMovingImage(TImage * inputImage)
{
  this->ProcessObject::SetInput("MovingImage", inputImage);
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetMovingImage() const -> const InputImageType *
{
  return itkDynamicCastInDebugMode<const TImage *>(this->ProcessObject::GetInput("MovingImage"));
}


template <typename TImage>
void
TransformixFilter<TImage>::RemoveMovingImage()
{
  this->ProcessObject::RemoveInput("MovingImage");
}

template <typename TImage>
void
TransformixFilter<TImage>::SetInput(InputImageType * inputImage)
{
  this->ProcessObject::SetInput("MovingImage", inputImage);
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetInput() const -> const InputImageType *
{
  return itkDynamicCastInDebugMode<const TImage *>(this->ProcessObject::GetInput("MovingImage"));
}

template <typename TImage>
const DataObject *
TransformixFilter<TImage>::GetInput(DataObjectPointerArraySizeType index) const
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


template <typename TImage>
void
TransformixFilter<TImage>::SetInput(DataObjectPointerArraySizeType index, DataObject * input)
{
  switch (index)
  {
    case 0:
      this->SetMovingImage(itkDynamicCastInDebugMode<TImage *>(input));
      break;
    case 1:
      this->SetTransformParameterObject(itkDynamicCastInDebugMode<ParameterObjectType *>(input));
      break;
    default:
      this->ProcessObject::SetNthInput(index, input);
  }
}

template <typename TImage>
void
TransformixFilter<TImage>::SetTransformParameterObject(ParameterObjectType * parameterObject)
{
  this->ProcessObject::SetInput("TransformParameterObject", parameterObject);
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetTransformParameterObject() -> ParameterObjectType *
{
  return itkDynamicCastInDebugMode<ParameterObjectType *>(this->ProcessObject::GetInput("TransformParameterObject"));
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetTransformParameterObject() const -> const ParameterObjectType *
{
  return itkDynamicCastInDebugMode<const ParameterObjectType *>(
    this->ProcessObject::GetInput("TransformParameterObject"));
}


template <typename TImage>
void
TransformixFilter<TImage>::SetTransformParameterFileName(std::string fileName)
{
  if (m_TransformParameterFileName != fileName)
  {
    const auto parameterObject = elx::ParameterObject::New();
    parameterObject->AddParameterFile(fileName);
    SetTransformParameterObject(parameterObject);
    m_TransformParameterFileName = std::move(fileName);
    this->Modified();
  }
}

template <typename TImage>
auto
TransformixFilter<TImage>::GetOutputDeformationField() -> OutputDeformationFieldType *
{
  return itkDynamicCastInDebugMode<OutputDeformationFieldType *>(
    this->itk::ProcessObject::GetOutput("ResultDeformationField"));
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetOutputDeformationField() const -> const OutputDeformationFieldType *
{
  return itkDynamicCastInDebugMode<const OutputDeformationFieldType *>(
    this->itk::ProcessObject::GetOutput("ResultDeformationField"));
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetOutput() -> OutputImageType *
{
  return static_cast<OutputImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetOutput() const -> const OutputImageType *
{
  return static_cast<const OutputImageType *>(this->ProcessObject::GetOutput(0));
}


template <typename TImage>
DataObject *
TransformixFilter<TImage>::GetOutput(unsigned int idx)
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TImage>
const DataObject *
TransformixFilter<TImage>::GetOutput(unsigned int idx) const
{
  return this->ProcessObject::GetOutput(idx);
}


template <typename TImage>
bool
TransformixFilter<TImage>::IsEmpty(const InputImageType * inputImage)
{
  if (!inputImage)
  {
    return true;
  }

  typename TImage::RegionType region = inputImage->GetLargestPossibleRegion();
  return region.GetNumberOfPixels() == 0;
}


template <typename TImage>
void
TransformixFilter<TImage>::SetLogFileName(std::string logFileName)
{
  m_LogFileName = logFileName;
  this->LogToFileOn();
}


template <typename TImage>
void
TransformixFilter<TImage>::RemoveLogFileName()
{
  m_LogFileName = "";
  this->LogToFileOff();
}


template <typename TImage>
void
TransformixFilter<TImage>::SetTransform(const TransformBase * const transform)
{
  if (transform)
  {
    if (m_Transform != transform)
    {
      m_ExternalTransform = nullptr;
      m_Transform = transform;
      this->Modified();
    }
  }
  else
  {
    m_ExternalTransform = nullptr;
    m_Transform = nullptr;
    this->Modified();
  }
}


template <typename TImage>
void
TransformixFilter<TImage>::SetExternalTransform(TransformType * const transform)
{
  if (transform)
  {
    if (m_ExternalTransform != transform)
    {
      m_Transform = nullptr;
      m_ExternalTransform = transform;
      this->Modified();
    }
  }
  else
  {
    m_ExternalTransform = nullptr;
    m_Transform = nullptr;
    this->Modified();
  }
}


template <typename TImage>
auto
TransformixFilter<TImage>::GetFirstElastixTransformBase() const -> const ElastixTransformBaseType *
{
  const auto * const transformContainer = m_TransformixMain->GetElastixBase().GetTransformContainer();

  if ((transformContainer != nullptr) && (!transformContainer->empty()))
  {
    return dynamic_cast<ElastixTransformBaseType *>(transformContainer->front().GetPointer());
  }
  return nullptr;
}


} // namespace itk

#endif
