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
#ifndef itkElastixRegistrationMethod_h
#define itkElastixRegistrationMethod_h

#include "itkImageSource.h"

#include "elxElastixMain.h"
#include "elxParameterObject.h"

/**
 * \class ElastixRegistrationMethod
 * \brief ITK Filter interface to the Elastix registration library.
 *
 * \ingroup Elastix
 */

namespace itk
{

template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT ElastixRegistrationMethod : public itk::ImageSource<TFixedImage>
{
public:
  /** Standard ITK typedefs. */
  typedef ElastixRegistrationMethod Self;
  typedef ImageSource<TFixedImage>  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElastixRegistrationMethod, ImageSource);

  /** Typedefs. */
  typedef elastix::ElastixMain                      ElastixMainType;
  typedef ElastixMainType::Pointer                  ElastixMainPointer;
  typedef std::vector<ElastixMainPointer>           ElastixMainVectorType;
  typedef ElastixMainType::ObjectPointer            ElastixMainObjectPointer;
  typedef ElastixMainType::ArgumentMapType          ArgumentMapType;
  typedef ArgumentMapType::value_type               ArgumentMapEntryType;
  typedef ElastixMainType::FlatDirectionCosinesType FlatDirectionCosinesType;

  typedef ElastixMainType::DataObjectContainerType      DataObjectContainerType;
  typedef ElastixMainType::DataObjectContainerPointer   DataObjectContainerPointer;
  typedef DataObjectContainerType::Iterator             DataObjectContainerIterator;
  typedef ProcessObject::DataObjectIdentifierType       DataObjectIdentifierType;
  typedef ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
  typedef ProcessObject::NameArray                      NameArrayType;

  typedef elastix::ParameterObject                      ParameterObjectType;
  typedef ParameterObjectType::ParameterMapType         ParameterMapType;
  typedef ParameterObjectType::ParameterMapVectorType   ParameterMapVectorType;
  typedef ParameterObjectType::ParameterValueVectorType ParameterValueVectorType;
  typedef ParameterObjectType::Pointer                  ParameterObjectPointer;
  typedef ParameterObjectType::ConstPointer             ParameterObjectConstPointer;

  static constexpr unsigned int FixedImageDimension = TFixedImage::ImageDimension;
  static constexpr unsigned int MovingImageDimension = TMovingImage::ImageDimension;

  typedef Image<unsigned char, FixedImageDimension>  FixedMaskType;
  typedef Image<unsigned char, MovingImageDimension> MovingMaskType;

  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;
  using ResultImageType = FixedImageType;

  /** Set/Add/Get/NumberOf fixed images. */
  virtual void
  SetFixedImage(TFixedImage * fixedImage);
  virtual void
  AddFixedImage(TFixedImage * fixedImage);
  const FixedImageType *
  GetFixedImage() const;
  const FixedImageType *
  GetFixedImage(const unsigned int index) const;
  unsigned int
  GetNumberOfFixedImages() const;

  /** Set/Add/Get/NumberOf moving images. */
  virtual void
  SetMovingImage(TMovingImage * movingImages);
  virtual void
  AddMovingImage(TMovingImage * movingImage);
  const MovingImageType *
  GetMovingImage() const;
  const MovingImageType *
  GetMovingImage(const unsigned int index) const;
  unsigned int
  GetNumberOfMovingImages() const;

  /** Set/Add/Get/Remove/NumberOf fixed masks. */
  virtual void
  AddFixedMask(FixedMaskType * fixedMask);
  virtual void
  SetFixedMask(FixedMaskType * fixedMask);
  const FixedMaskType *
  GetFixedMask() const;
  const FixedMaskType *
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
  const MovingMaskType *
  GetMovingMask(void) const;
  const MovingMaskType *
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
  using Superclass::GetOutput;
  DataObject *
  GetOutput(unsigned int idx);
  const DataObject *
  GetOutput(unsigned int idx) const;
  ResultImageType *
  GetOutput();
  const ResultImageType *
  GetOutput() const;

  /* Standard filter indexed input / output methods */
  void
  SetInput(FixedImageType * fixedImage);
  const FixedImageType *
  GetInput() const;
  void
  SetInput(DataObjectPointerArraySizeType index, DataObject * input);
  const DataObject *
  GetInput(DataObjectPointerArraySizeType index) const;

  /** Set/Get/Remove initial transform parameter filename. */
  itkSetMacro(InitialTransformParameterFileName, std::string);
  itkGetMacro(InitialTransformParameterFileName, std::string);
  virtual void
  RemoveInitialTransformParameterFileName()
  {
    this->SetInitialTransformParameterFileName("");
  }

  /** Set/Get/Remove fixed point set filename. */
  itkSetMacro(FixedPointSetFileName, std::string);
  itkGetMacro(FixedPointSetFileName, std::string);
  void
  RemoveFixedPointSetFileName()
  {
    this->SetFixedPointSetFileName("");
  }

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro(MovingPointSetFileName, std::string);
  itkGetMacro(MovingPointSetFileName, std::string);
  void
  RemoveMovingPointSetFileName()
  {
    this->SetMovingPointSetFileName("");
  }

  /** Set/Get/Remove output directory. */
  itkSetMacro(OutputDirectory, std::string);
  itkGetMacro(OutputDirectory, std::string);
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

  itkSetMacro(NumberOfThreads, int);
  itkGetMacro(NumberOfThreads, int);

protected:
  ElastixRegistrationMethod();

  void
  GenerateData() override;

  using DataObjectPointer = ProcessObject::DataObjectPointer;
  using Superclass::MakeOutput;
  DataObjectPointer
  MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
  ElastixRegistrationMethod(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented

  /** MakeUniqueName. */
  std::string
  MakeUniqueName(const DataObjectIdentifierType & key);

  /** IsInputOfType. */
  bool
  IsInputOfType(const DataObjectIdentifierType & InputOfType, DataObjectIdentifierType inputName) const;

  /** GetNumberOfInputsOfType */
  unsigned int
  GetNumberOfInputsOfType(const DataObjectIdentifierType & intputType) const;

  /** RemoveInputsOfType. */
  void
  RemoveInputsOfType(const DataObjectIdentifierType & inputName);

  std::string m_InitialTransformParameterFileName;
  std::string m_FixedPointSetFileName;
  std::string m_MovingPointSetFileName;

  std::string m_OutputDirectory;
  std::string m_LogFileName;

  bool m_LogToConsole;
  bool m_LogToFile;

  int m_NumberOfThreads;

  unsigned int m_InputUID;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkElastixRegistrationMethod.hxx"
#endif

#endif // itkElastixRegistrationMethod_h
