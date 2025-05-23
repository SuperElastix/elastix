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
#ifndef itkTransformixFilter_h
#define itkTransformixFilter_h

#include "itkElastixLogLevel.h"
#include "itkImageSource.h"
#include "itkMesh.h"
#include "itkTransformBase.h"

#include "elxElastixTemplate.h"
#include "elxTransformBase.h"
#include "elxTransformixMain.h"
#include "elxParameterObject.h"

/**
 * \class TransformixFilter
 * \brief ITK Filter interface to the Transformix library.
 *
 * \ingroup Elastix
 */

namespace itk
{

template <typename TImage>
class ITK_TEMPLATE_EXPORT TransformixFilter : public ImageSource<TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(TransformixFilter);

  /** Standard ITK typedefs. */
  using Self = TransformixFilter;
  using Superclass = ImageSource<TImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(TransformixFilter);

  /** Typedefs. */
  using TransformixMainType = elx::TransformixMain;
  using TransformixMainPointer = TransformixMainType::Pointer;
  using ArgumentMapType = TransformixMainType::ArgumentMapType;
  using ArgumentMapEntryType = ArgumentMapType::value_type;

  using DataObjectPointer = ProcessObject::DataObjectPointer;
  using DataObjectIdentifierType = ProcessObject::DataObjectIdentifierType;
  using DataObjectContainerType = TransformixMainType::DataObjectContainerType;
  using DataObjectContainerPointer = TransformixMainType::DataObjectContainerPointer;

  using ParameterObjectType = elx::ParameterObject;
  using ParameterMapVectorType = ParameterObjectType::ParameterMapVectorType;
  using ParameterMapType = ParameterObjectType::ParameterMapType;
  using ParameterValueVectorType = ParameterObjectType::ParameterValueVectorType;
  using ParameterObjectPointer = typename ParameterObjectType::Pointer;
  using ParameterObjectConstPointer = typename ParameterObjectType::ConstPointer;

  using typename Superclass::OutputImageType;
  using typename Superclass::OutputImagePixelType;

  using OutputDeformationFieldType =
    typename itk::Image<itk::Vector<float, TImage::ImageDimension>, TImage::ImageDimension>;

  using DataObjectPointerArraySizeType = ProcessObject::DataObjectPointerArraySizeType;

  using InputImageType = TImage;
  itkStaticConstMacro(MovingImageDimension, unsigned int, TImage::ImageDimension);

  using MeshType = Mesh<OutputImagePixelType, MovingImageDimension>;

  using TransformType = Transform<double, MovingImageDimension, MovingImageDimension>;

  /** Typedefs for images of determinants of spatial Jacobian matrices, and images of spatial Jacobian matrices */
  using SpatialJacobianDeterminantImageType = itk::Image<float, MovingImageDimension>;
  using SpatialJacobianMatrixImageType =
    itk::Image<itk::Matrix<float, MovingImageDimension, MovingImageDimension>, MovingImageDimension>;

  /** Set/Get/Add moving image. */
  virtual void
  SetMovingImage(TImage * inputImage);
  const InputImageType *
  GetMovingImage() const;
  virtual void
  RemoveMovingImage();

  /* Standard filter indexed input / output methods */
  void
  SetInput(InputImageType * movingImage);
  const InputImageType *
  GetInput() const;
  void
  SetInput(DataObjectPointerArraySizeType index, DataObject * input);
  const DataObject *
  GetInput(DataObjectPointerArraySizeType index) const;

  void SetTransformParameterFileName(std::string);
  itkGetConstMacro(TransformParameterFileName, std::string);

  /** Set/Get/Remove fixed point set filename. */
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
  SetTransformParameterObject(ParameterObjectType * transformParameterObject);

  ParameterObjectType *
  GetTransformParameterObject();

  const ParameterObjectType *
  GetTransformParameterObject() const;

  using Superclass::GetOutput;
  DataObject *
  GetOutput(unsigned int idx);
  const DataObject *
  GetOutput(unsigned int idx) const;
  OutputImageType *
  GetOutput();
  const OutputImageType *
  GetOutput() const;

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

  itkSetMacro(LogLevel, ElastixLogLevel);
  itkGetConstMacro(LogLevel, ElastixLogLevel);

  /** Sets an (optional) input mesh. An Update() will transform its points, and store them in the output mesh.  */
  itkSetConstObjectMacro(InputMesh, MeshType);

  /** Retrieves the output mesh, produced by an Update(), when an input mesh was specified.  */
  const MeshType *
  GetOutputMesh() const
  {
    return m_OutputMesh;
  }

  /** Sets the transformation. If null, the transformation is entirely specified by the transform parameter object that
   * is set by SetTransformParameterObject. Otherwise, the transformation is specified by this transform object, with
   * additional information from the specified transform parameter object. Will override a possible previous call to
   * SetExternalTransform. */
  void
  SetTransform(const TransformBase *);

  /** Set the transformation by means of an external ITK Transform. If null, the transformation is entirely specified by
   * the transform parameter object that is set by SetTransformParameterObject. Otherwise, the transformation is
   * specified by this transform object, with additional information from the specified transform parameter object. Will
   * override a possible previous call to SetTransform. */
  void
  SetExternalTransform(TransformType *);

  itkSetObjectMacro(CombinationTransform, TransformType);

  /** Computes the spatial Jacobian determinant for each pixel, and returns an image of the computed values.
  \note Before calling this member function, Update() must be called. */
  SmartPointer<SpatialJacobianDeterminantImageType>
  ComputeSpatialJacobianDeterminantImage() const;

  /** Computes the spatial Jacobian matrix for each pixel, and returns an image of the computed matrices.
  \note Before calling this member function, Update() must be called. */
  SmartPointer<SpatialJacobianMatrixImageType>
  ComputeSpatialJacobianMatrixImage() const;

protected:
  TransformixFilter();

  /** To support outputs of different types (i.e. ResultImage and ResultDeformationField)
   * MakeOutput from itk::ImageSource< TOutputImage > needs to be overridden.
   */
  DataObjectPointer
  MakeOutput(const DataObjectIdentifierType & key) override;

  /** The ResultImage and ResultDeformationField get their image properties from the TransformParameterObject. */
  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

private:
  /** Private using-declarations, just to avoid GCC compilation warnings: '...' was hidden [-Woverloaded-virtual] */
  using Superclass::SetInput;
  using Superclass::MakeOutput;

  /** IsEmpty. */
  static bool
  IsEmpty(const InputImageType * inputImage);

  /** Tell the compiler we want all definitions of Get/Set/Remove
   *  from ProcessObject and TransformixFilter.
   */
  using ProcessObject::RemoveInput;

  using ElastixTransformBaseType = elx::TransformBase<elx::ElastixTemplate<TImage, TImage>>;

  const ElastixTransformBaseType *
  GetFirstElastixTransformBase() const;

  SmartPointer<const elx::TransformixMain> m_TransformixMain{};

  std::string m_TransformParameterFileName{};
  std::string m_FixedPointSetFileName{};
  bool        m_ComputeSpatialJacobian{ false };
  bool        m_ComputeDeterminantOfSpatialJacobian{ false };
  bool        m_ComputeDeformationField{ false };

  std::string m_OutputDirectory{};
  std::string m_LogFileName{};

  bool m_EnableOutput{ true };
  bool m_LogToConsole{ false };
  bool m_LogToFile{ false };

  ElastixLogLevel m_LogLevel{};

  SmartPointer<const MeshType> m_InputMesh{};
  SmartPointer<MeshType>       m_OutputMesh{};

  SmartPointer<const TransformBase> m_Transform{};
  SmartPointer<TransformType>       m_ExternalTransform{};

  SmartPointer<TransformType> m_CombinationTransform;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTransformixFilter.hxx"
#endif

#endif
