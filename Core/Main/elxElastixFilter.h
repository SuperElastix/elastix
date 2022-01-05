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
#ifndef elxElastixFilter_h
#define elxElastixFilter_h

#include "itkImageSource.h"

#include "elxElastixMain.h"
#include "elxParameterObject.h"
#include "elxPixelType.h"

/**
 * \class ElastixFilter
 * \brief ITK Filter interface to the Elastix registration library.
 */

namespace elastix
{

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT ELASTIXLIB_API ElastixFilter : public itk::ImageSource<TFixedImage>
{
public:
  /** Standard ITK typedefs. */
  using Self = ElastixFilter;
  using Superclass = itk::ImageSource<TFixedImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElastixFilter, itk::ImageSource);

  /** Typedefs. */
  using ElastixMainType = elastix::ElastixMain;
  using ElastixMainPointer = ElastixMainType::Pointer;
  using ElastixMainVectorType = std::vector<ElastixMainPointer>;
  using ElastixMainObjectPointer = ElastixMainType::ObjectPointer;
  using ArgumentMapType = ElastixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;
  using FlatDirectionCosinesType = ElastixMainType::FlatDirectionCosinesType;

  using DataObjectContainerType = ElastixMainType::DataObjectContainerType;
  using DataObjectContainerPointer = ElastixMainType::DataObjectContainerPointer;
  using DataObjectContainerIterator = DataObjectContainerType::Iterator;
  using DataObjectIdentifierType = itk::ProcessObject::DataObjectIdentifierType;
  using DataObjectPointerArraySizeType = itk::ProcessObject::DataObjectPointerArraySizeType;
  using NameArrayType = itk::ProcessObject::NameArray;

  using ParameterObjectType = ParameterObject;
  using ParameterMapType = ParameterObjectType::ParameterMapType;
  using ParameterMapVectorType = ParameterObjectType::ParameterMapVectorType;
  using ParameterValueVectorType = ParameterObjectType::ParameterValueVectorType;
  using ParameterObjectPointer = ParameterObjectType::Pointer;
  using ParameterObjectConstPointer = ParameterObjectType::ConstPointer;

  using FixedImagePointer = typename TFixedImage::Pointer;
  using FixedImageConstPointer = typename TFixedImage::ConstPointer;
  using MovingImagePointer = typename TMovingImage::Pointer;
  using MovingImageConstPointer = typename TMovingImage::ConstPointer;

  itkStaticConstMacro(FixedImageDimension, unsigned int, TFixedImage::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, TMovingImage::ImageDimension);

  using FixedMaskType = itk::Image<unsigned char, FixedImageDimension>;
  using FixedMaskPointer = typename FixedMaskType::Pointer;
  using FixedMaskConstPointer = typename FixedMaskType::Pointer;
  using MovingMaskType = itk::Image<unsigned char, MovingImageDimension>;
  using MovingMaskPointer = typename MovingMaskType::Pointer;
  using MovingMaskConstPointer = typename MovingMaskType::Pointer;

  /** Set/Add/Get/NumberOf fixed images. */
  virtual void
  SetFixedImage(TFixedImage * fixedImage);
  virtual void
  AddFixedImage(TFixedImage * fixedImage);
  FixedImageConstPointer
  GetFixedImage() const;
  FixedImageConstPointer
  GetFixedImage(const unsigned int index) const;
  unsigned int
  GetNumberOfFixedImages() const;

  /** Set/Add/Get/NumberOf moving images. */
  virtual void
  SetMovingImage(TMovingImage * movingImages);
  virtual void
  AddMovingImage(TMovingImage * movingImage);
  MovingImageConstPointer
  GetMovingImage() const;
  MovingImageConstPointer
  GetMovingImage(const unsigned int index) const;
  unsigned int
  GetNumberOfMovingImages() const;

  /** Set/Add/Get/Remove/NumberOf fixed masks. */
  virtual void
  AddFixedMask(FixedMaskType * fixedMask);
  virtual void
  SetFixedMask(FixedMaskType * fixedMask);
  FixedMaskConstPointer
  GetFixedMask() const;
  FixedMaskConstPointer
  GetFixedMask(const unsigned int index) const;
  void
  RemoveFixedMask();
  unsigned int
  GetNumberOfFixedMasks() const;

  /** Set/Add/Get/Remove/NumberOf moving masks. */
  virtual void
  SetMovingMask(MovingMaskType * movingMask);
  virtual void
  AddMovingMask(MovingMaskType * movingMask);
  MovingMaskConstPointer
  GetMovingMask() const;
  MovingMaskConstPointer
  GetMovingMask(const unsigned int index) const;
  virtual void
  RemoveMovingMask();
  unsigned int
  GetNumberOfMovingMasks() const;

  /** Set/Get parameter object.*/
  virtual void
  SetParameterObject(ParameterObjectType * parameterObject);
  ParameterObjectType *
  GetParameterObject();
  const ParameterObjectType *
  GetParameterObject() const;

  /** Get transform parameter object.*/
  ParameterObjectType *
  GetTransformParameterObject();
  const ParameterObjectType *
  GetTransformParameterObject() const;

  /** Set/Get/Remove initial transform parameter filename. */
  itkSetMacro(InitialTransformParameterFileName, std::string);
  itkGetConstMacro(InitialTransformParameterFileName, std::string);
  virtual void
  RemoveInitialTransformParameterFileName()
  {
    this->SetInitialTransformParameterFileName("");
  }

  /** Set/Get/Remove fixed point set filename. */
  itkSetMacro(FixedPointSetFileName, std::string);
  itkGetConstMacro(FixedPointSetFileName, std::string);
  void
  RemoveFixedPointSetFileName()
  {
    this->SetFixedPointSetFileName("");
  }

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro(MovingPointSetFileName, std::string);
  itkGetConstMacro(MovingPointSetFileName, std::string);
  void
  RemoveMovingPointSetFileName()
  {
    this->SetMovingPointSetFileName("");
  }

  /** Set/Get/Remove output directory. */
  itkSetMacro(OutputDirectory, std::string);
  itkGetConstMacro(OutputDirectory, std::string);
  void
  RemoveOutputDirectory()
  {
    this->SetOutputDirectory("");
  }

  /** Set/Get/Remove log filename. */
  void
  SetLogFileName(const std::string logFileName);

  itkGetConstMacro(LogFileName, std::string);
  void
  RemoveLogFileName();

  /** Log to std::cout on/off. */
  itkSetMacro(LogToConsole, bool);
  itkGetConstReferenceMacro(LogToConsole, bool);
  itkBooleanMacro(LogToConsole);

  /** Log to file on/off. */
  itkSetMacro(LogToFile, bool);
  itkGetConstReferenceMacro(LogToFile, bool);
  itkBooleanMacro(LogToFile);

  /** Disables output to log and standard output. */
  void
  DisableOutput()
  {
    m_EnableOutput = false;
  }

  itkSetMacro(NumberOfThreads, int);
  itkGetConstMacro(NumberOfThreads, int);

protected:
  ElastixFilter();

  virtual void
  GenerateData() override;

private:
  ElastixFilter(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** MakeUniqueName. */
  std::string
  MakeUniqueName(const DataObjectIdentifierType & key);

  /** IsInputOfType. */
  bool
  IsInputOfType(const DataObjectIdentifierType & InputOfType, const DataObjectIdentifierType & inputName);

  /** GetNumberOfInputsOfType */
  unsigned int
  GetNumberOfInputsOfType(const DataObjectIdentifierType & intputType);

  /** RemoveInputsOfType. */
  void
  RemoveInputsOfType(const DataObjectIdentifierType & inputName);

  std::string m_InitialTransformParameterFileName;
  std::string m_FixedPointSetFileName;
  std::string m_MovingPointSetFileName;

  std::string m_OutputDirectory;
  std::string m_LogFileName;

  bool m_EnableOutput{ true };
  bool m_LogToConsole;
  bool m_LogToFile;

  int m_NumberOfThreads;

  unsigned int m_InputUID;
};

} // namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxElastixFilter.hxx"
#endif

#endif // elxElastixFilter_h
