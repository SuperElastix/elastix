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
#include "itkAdvancedCombinationTransform.h"
#include "itkElastixLogLevel.h"

#include "elxElastixMain.h"
#include "elxElastixTemplate.h"
#include "elxElastixBase.h"
#include "elxTransformBase.h"
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
  ITK_DISALLOW_COPY_AND_MOVE(ElastixRegistrationMethod);

  /** Standard ITK typedefs. */
  using Self = ElastixRegistrationMethod;
  using Superclass = ImageSource<TFixedImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ElastixRegistrationMethod);

  /** Typedefs. */
  using ElastixMainType = elx::ElastixMain;
  using ElastixMainPointer = ElastixMainType::Pointer;
  using ElastixMainVectorType = std::vector<ElastixMainPointer>;
  using ElastixMainObjectPointer = ElastixMainType::ObjectPointer;
  using ArgumentMapType = ElastixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;
  using FlatDirectionCosinesType = ElastixMainType::FlatDirectionCosinesType;

  using DataObjectContainerType = ElastixMainType::DataObjectContainerType;
  using DataObjectContainerPointer = ElastixMainType::DataObjectContainerPointer;
  using DataObjectContainerIterator = DataObjectContainerType::Iterator;
  using DataObjectIdentifierType = ProcessObject::DataObjectIdentifierType;
  using DataObjectPointerArraySizeType = ProcessObject::DataObjectPointerArraySizeType;
  using NameArrayType = ProcessObject::NameArray;

  using ParameterObjectType = elx::ParameterObject;
  using ParameterMapType = ParameterObjectType::ParameterMapType;
  using ParameterMapVectorType = ParameterObjectType::ParameterMapVectorType;
  using ParameterValueVectorType = ParameterObjectType::ParameterValueVectorType;
  using ParameterObjectPointer = ParameterObjectType::Pointer;
  using ParameterObjectConstPointer = ParameterObjectType::ConstPointer;

  static constexpr unsigned int FixedImageDimension = TFixedImage::ImageDimension;
  static constexpr unsigned int MovingImageDimension = TMovingImage::ImageDimension;

  static_assert(FixedImageDimension == MovingImageDimension,
                "ElastixRegistrationMethod assumes that fixed and moving image have the same number of dimensions.");
  static constexpr unsigned int ImageDimension = TFixedImage::ImageDimension;

  template <typename TCoordinate>
  using PointContainerType = VectorContainer<IdentifierType, Point<TCoordinate, ImageDimension>>;

  using FixedMaskType = Image<unsigned char, FixedImageDimension>;
  using MovingMaskType = Image<unsigned char, MovingImageDimension>;
  using TransformType = Transform<double, FixedImageDimension, MovingImageDimension>;

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
  GetMovingMask() const;
  const MovingMaskType *
  GetMovingMask(const unsigned int index) const;
  virtual void
  RemoveMovingMask();
  unsigned int
  GetNumberOfMovingMasks() const;

  itkSetConstObjectMacro(FixedPoints, PointContainerType<double>);
  itkSetConstObjectMacro(MovingPoints, PointContainerType<double>);

  void
  SetFixedPoints(const PointContainerType<float> * const points)
  {
    SetFixedPoints(ConvertToPointContainerOfDoubleCoordinates(points));
  }

  void
  SetMovingPoints(const PointContainerType<float> * const points)
  {
    SetMovingPoints(ConvertToPointContainerOfDoubleCoordinates(points));
  }

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

  /* \note When "WriteResultImage" is false, the output image will be empty. */
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
  void SetInitialTransformParameterFileName(std::string);

  itkGetConstMacro(InitialTransformParameterFileName, std::string);
  virtual void
  RemoveInitialTransformParameterFileName()
  {
    this->SetInitialTransformParameterFileName("");
  }

  /** Set initial transform parameter object. */
  void
  SetInitialTransformParameterObject(const elx::ParameterObject *);

  /** Returns the previously specified initial ITK Transform. Only allows const access to the transform. */
  const TransformType *
  GetInitialTransform() const
  {
    return m_InitialTransform;
  }

  /** Set the initial transformation by means of an ITK Transform. */
  void
  SetInitialTransform(const TransformType *);

  /** Returns the previously specified external ITK Transform. Note that it allows full access to the transform, even
   * when having const-only access to this ElastixRegistrationMethod, because the transform is external to the
   * ElastixRegistrationMethod object. */
  TransformType *
  GetExternalInitialTransform() const
  {
    return m_ExternalInitialTransform;
  }

  /** Set the initial transformation by means of an external ITK Transform. */
  void
  SetExternalInitialTransform(TransformType *);

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

  itkSetMacro(LogLevel, ElastixLogLevel);
  itkGetConstMacro(LogLevel, ElastixLogLevel);

  itkSetMacro(NumberOfThreads, int);
  itkGetConstMacro(NumberOfThreads, int);

  /** Returns the number of transformations, produced during the last Update(). */
  unsigned int
  GetNumberOfTransforms() const;

  /** Returns the nth transformation, produced during the last Update(). */
  TransformType *
  GetNthTransform(const unsigned int n) const;

  /** Returns the combination transformation, produced during the last Update(). */
  TransformType *
  GetCombinationTransform() const;

  /** Converts the specified elastix Transform object to the corresponding ITK Transform object. Returns null if there
   * is no corresponding ITK Transform type. */
  static SmartPointer<TransformType>
  ConvertToItkTransform(const TransformType &);

protected:
  ElastixRegistrationMethod();

  void
  GenerateData() override;

  using DataObjectPointer = ProcessObject::DataObjectPointer;
  using Superclass::MakeOutput;
  DataObjectPointer
  MakeOutput(DataObjectPointerArraySizeType idx) override;

private:
  static SmartPointer<PointContainerType<double>>
  ConvertToPointContainerOfDoubleCoordinates(const PointContainerType<float> * const floatPointContainer)
  {
    if (floatPointContainer)
    {
      const auto result = PointContainerType<double>::New();
      result->assign(floatPointContainer->cbegin(), floatPointContainer->cend());
      return result;
    }
    else
    {
      return nullptr;
    }
  }

  /** MakeUniqueName. */
  std::string
  MakeUniqueName(const DataObjectIdentifierType & key);

  /** IsInputOfType. */
  bool
  IsInputOfType(const DataObjectIdentifierType & InputOfType, const DataObjectIdentifierType & inputName) const;

  /** GetNumberOfInputsOfType */
  unsigned int
  GetNumberOfInputsOfType(const DataObjectIdentifierType & intputType) const;

  /** RemoveInputsOfType. */
  void
  RemoveInputsOfType(const DataObjectIdentifierType & inputName);


  /** Retrieves either the fixed or the moving input images. */
  template <typename TImage>
  std::vector<TImage *>
  GetInputImages(const char * const inputTypeString)
  {
    std::vector<TImage *> images;
    for (const auto & inputName : this->GetInputNames())
    {
      if (this->IsInputOfType(inputTypeString, inputName))
      {
        images.push_back(itkDynamicCastInDebugMode<TImage *>(this->ProcessObject::GetInput(inputName)));
      }
    }
    return images;
  }

  void
  ResetInitialTransformWithoutModified()
  {
    m_InitialTransform = nullptr;
    m_InitialTransformParameterFileName.clear();
    m_InitialTransformParameterObject = nullptr;
    m_ExternalInitialTransform = nullptr;
  }

  void
  ResetInitialTransformAndModified()
  {
    if (m_InitialTransform || m_InitialTransformParameterObject || m_ExternalInitialTransform ||
        !m_InitialTransformParameterFileName.empty())
    {
      ResetInitialTransformWithoutModified();
      this->Modified();
    }
  }

  /** Private using-declaration, just to avoid GCC compilation warnings: '...' was hidden [-Woverloaded-virtual] */
  using Superclass::SetInput;

  using AdvancedCombinationTransformType =
    AdvancedCombinationTransform<elx::ElastixBase::CoordinateType, FixedImageDimension>;

  AdvancedCombinationTransformType *
  GetAdvancedCombinationTransform() const;

  SmartPointer<const elx::ElastixMain> m_ElastixMain{};

  std::string                              m_InitialTransformParameterFileName{};
  SmartPointer<const elx::ParameterObject> m_InitialTransformParameterObject{};
  SmartPointer<const TransformType>        m_InitialTransform{};
  SmartPointer<TransformType>              m_ExternalInitialTransform{};

  std::string m_FixedPointSetFileName{};
  std::string m_MovingPointSetFileName{};

  SmartPointer<const PointContainerType<double>> m_FixedPoints{};
  SmartPointer<const PointContainerType<double>> m_MovingPoints{};

  std::string m_OutputDirectory{};
  std::string m_LogFileName{};

  bool m_EnableOutput{ true };
  bool m_LogToConsole{ false };
  bool m_LogToFile{ false };

  ElastixLogLevel m_LogLevel{};

  int m_NumberOfThreads{ 0 };

  unsigned int m_InputUID{ 0 };
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkElastixRegistrationMethod.hxx"
#endif

#endif // itkElastixRegistrationMethod_h
