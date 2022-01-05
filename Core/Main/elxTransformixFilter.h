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
#ifndef elxTransformixFilter_h
#define elxTransformixFilter_h

#include "itkImageSource.h"

#include "elxTransformixMain.h"
#include "elxParameterObject.h"
#include "elxPixelType.h"

/**
 * \class TransformixFilter
 * \brief ITK Filter interface to the Transformix library.
 */

namespace elastix
{

template <typename TMovingImage>
class ITK_TEMPLATE_EXPORT ELASTIXLIB_API TransformixFilter : public itk::ImageSource<TMovingImage>
{
public:
  /** Standard ITK typedefs. */
  using Self = TransformixFilter;
  using Superclass = itk::ImageSource<TMovingImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TransformixFilter, itk::ImageSource);

  /** Typedefs. */
  using TransformixMainType = elastix::TransformixMain;
  using TransformixMainPointer = TransformixMainType::Pointer;
  using ArgumentMapType = TransformixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;

  using DataObjectPointer = itk::ProcessObject::DataObjectPointer;
  using DataObjectIdentifierType = itk::ProcessObject::DataObjectIdentifierType;
  using DataObjectContainerType = TransformixMainType::DataObjectContainerType;
  using DataObjectContainerPointer = TransformixMainType::DataObjectContainerPointer;

  using ParameterObjectType = ParameterObject;
  using ParameterMapVectorType = ParameterObjectType::ParameterMapVectorType;
  using ParameterMapType = ParameterObjectType::ParameterMapType;
  using ParameterValueVectorType = ParameterObjectType::ParameterValueVectorType;
  using ParameterObjectPointer = typename ParameterObjectType::Pointer;
  using ParameterObjectConstPointer = typename ParameterObjectType::ConstPointer;

  using typename Superclass::OutputImageType;
  using OutputDeformationFieldType =
    typename itk::Image<itk::Vector<float, TMovingImage::ImageDimension>, TMovingImage::ImageDimension>;

  using InputImagePointer = typename TMovingImage::Pointer;
  using InputImageConstPointer = typename TMovingImage::ConstPointer;

  itkStaticConstMacro(MovingImageDimension, unsigned int, TMovingImage::ImageDimension);

  /** Set/Get/Add moving image. */
  virtual void
  SetMovingImage(TMovingImage * inputImage);
  InputImageConstPointer
  GetMovingImage();
  virtual void
  RemoveMovingImage();

  /** Set/Get/Remove moving point set filename. */
  itkSetMacro(FixedPointSetFileName, std::string);
  itkGetConstMacro(FixedPointSetFileName, std::string);
  virtual void
  RemoveFixedPointSetFileName()
  {
    this->SetFixedPointSetFileName("");
  }

  /** Compute spatial Jacobian On/Off. */
  itkSetMacro(ComputeSpatialJacobian, bool);
  itkGetConstMacro(ComputeSpatialJacobian, bool);
  itkBooleanMacro(ComputeSpatialJacobian);

  /** Compute determinant of spatial Jacobian On/Off. */
  itkSetMacro(ComputeDeterminantOfSpatialJacobian, bool);
  itkGetConstMacro(ComputeDeterminantOfSpatialJacobian, bool);
  itkBooleanMacro(ComputeDeterminantOfSpatialJacobian);

  /** Compute deformation field On/Off. */
  itkSetMacro(ComputeDeformationField, bool);
  itkGetConstMacro(ComputeDeformationField, bool);
  itkBooleanMacro(ComputeDeformationField);

  /** Get/Set transform parameter object. */
  virtual void
  SetTransformParameterObject(ParameterObjectPointer transformParameterObject);

  ParameterObjectType *
  GetTransformParameterObject();

  const ParameterObjectType *
  GetTransformParameterObject() const;

  OutputDeformationFieldType *
  GetOutputDeformationField();

  const OutputDeformationFieldType *
  GetOutputDeformationField() const;

  /** Set/Get/Remove output directory. */
  itkSetMacro(OutputDirectory, std::string);
  itkGetConstMacro(OutputDirectory, std::string);
  virtual void
  RemoveOutputDirectory()
  {
    this->SetOutputDirectory("");
  }

  /** Set/Get/Remove log filename. */
  virtual void
  SetLogFileName(std::string logFileName);

  itkGetConstMacro(LogFileName, std::string);
  virtual void
  RemoveLogFileName();

  /** Log to std::cout on/off. */
  itkSetMacro(LogToConsole, bool);
  itkGetConstMacro(LogToConsole, bool);
  itkBooleanMacro(LogToConsole);

  /** Log to file on/off. */
  itkSetMacro(LogToFile, bool);
  itkGetConstMacro(LogToFile, bool);
  itkBooleanMacro(LogToFile);

  /** Disables output to log and standard output. */
  void
  DisableOutput()
  {
    m_EnableOutput = false;
  }

  /** To support outputs of different types (i.e. ResultImage and ResultDeformationField)
   * MakeOutput from itk::ImageSource< TOutputImage > needs to be overridden.
   */
  virtual DataObjectPointer
  MakeOutput(const DataObjectIdentifierType & key) override;

  /** The ResultImage and ResultDeformationField get their image properties from the TransformParameterObject. */
  virtual void
  GenerateOutputInformation() override;

protected:
  TransformixFilter();

  virtual void
  GenerateData() override;

private:
  TransformixFilter(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** IsEmpty. */
  static bool
  IsEmpty(const InputImagePointer inputImage);

  /** Tell the compiler we want all definitions of Get/Set/Remove
   *  from ProcessObject and TransformixFilter.
   */
  using itk::ProcessObject::SetInput;
  using itk::ProcessObject::GetInput;
  using itk::ProcessObject::RemoveInput;

  std::string m_FixedPointSetFileName;
  bool        m_ComputeSpatialJacobian;
  bool        m_ComputeDeterminantOfSpatialJacobian;
  bool        m_ComputeDeformationField;

  std::string m_OutputDirectory;
  std::string m_LogFileName;

  bool m_EnableOutput{ true };
  bool m_LogToConsole;
  bool m_LogToFile;
};

} // namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxTransformixFilter.hxx"
#endif

#endif // elxTransformixFilter_h
