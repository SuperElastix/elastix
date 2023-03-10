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
#ifndef elxElastixMain_h
#define elxElastixMain_h

#include "elxMainBase.h"


namespace elastix
{

/**
 * \class ElastixMain
 * \brief A class with all functionality to configure elastix.
 *
 * The ElastixMain initializes the MyConfiguration class with the
 * parameters and commandline arguments. After this, the class loads
 * and creates all components and sets them in ElastixTemplate.
 *
 * \parameter FixedImageDimension: the dimension of the fixed image. \n
 * example: <tt>(FixedImageDimension 2)</tt>\n
 * \parameter MovingImageDimension: the dimension of the fixed image. \n
 * example: <tt>(MovingImageDimension 2)</tt>\n
 * \parameter FixedInternalImagePixelType: the pixel type of the internal
 * fixed image representation. The fixed image is automatically converted
 * to this type.\n
 * example: <tt>(FixedInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 * \parameter MovingInternalImagePixelType: the pixel type of the internal
 * moving image representation. The moving image is automatically converted
 * to this type.\n
 * example: <tt>(MovingInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 *
 * \transformparameter FixedImageDimension: the dimension of the fixed image. \n
 * example: <tt>(FixedImageDimension 2)</tt>\n
 * \transformparameter MovingImageDimension: the dimension of the fixed image. \n
 * example: <tt>(MovingImageDimension 2)</tt>\n
 * \transformparameter FixedInternalImagePixelType: the pixel type of the internal
 * fixed image representation. The fixed image is automatically converted
 * to this type.\n
 * example: <tt>(FixedInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 * \transformparameter MovingInternalImagePixelType: the pixel type of the internal
 * moving image representation. The moving image is automatically converted
 * to this type.\n
 * example: <tt>(MovingInternalImagePixelType "float")</tt>\n
 * Default/recommended: "float"\n
 *
 * \ingroup Kernel
 */

class ElastixMain : public MainBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ElastixMain);

  /** Standard itk. */
  using Self = ElastixMain;
  using Superclass = MainBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElastixMain, MainBase);

  /** Set/Get functions for the fixed images
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(FixedImageContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(FixedImageContainer, DataObjectContainerType);

  /** Set/Get functions for the fixed and moving masks
   * (if these are not used, elastix tries to read them from disk,
   * according to the command line parameters).
   */
  itkSetObjectMacro(FixedMaskContainer, DataObjectContainerType);
  itkSetObjectMacro(MovingMaskContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(FixedMaskContainer, DataObjectContainerType);
  itkGetModifiableObjectMacro(MovingMaskContainer, DataObjectContainerType);

  /** Get the final transform (the result of running elastix).
   * You may pass this as an InitialTransform in an other instantiation
   * of ElastixMain.
   * Only valid after calling Run()!
   */
  itkGetModifiableObjectMacro(FinalTransform, itk::Object);

  /** Set/Get the initial transform
   * the type is itk::Object, but the pointer should actually point
   * to an itk::Transform type (or inherited from that one).
   */
  itkSetObjectMacro(InitialTransform, itk::Object);
  itkGetModifiableObjectMacro(InitialTransform, itk::Object);

  /** Set/Get the original fixed image direction as a flat array
   * (d11 d21 d31 d21 d22 etc ) */
  virtual void
  SetOriginalFixedImageDirectionFlat(const FlatDirectionCosinesType & arg);

  virtual const FlatDirectionCosinesType &
  GetOriginalFixedImageDirectionFlat() const;

  /** Get and Set the elastix level. */
  void
  SetElastixLevel(unsigned int level);

  unsigned int
  GetElastixLevel() const;

  /** Get and Set the total number of elastix levels. */
  void
  SetTotalNumberOfElastixLevels(unsigned int levels);

  unsigned int
  GetTotalNumberOfElastixLevels() const;

  using Superclass::Run;

  /** Start the registration
   * run() without command line parameters; it assumes that
   * EnterCommandLineParameters has been invoked already, or that
   * m_Configuration is initialized in a different way.
   */
  int
  Run() override;

  /** GetTransformParametersMap */
  virtual ParameterMapType
  GetTransformParametersMap() const;

protected:
  ElastixMain();
  ~ElastixMain() override;

  /** The fixed images and masks. */
  DataObjectContainerPointer m_FixedImageContainer{ nullptr };
  DataObjectContainerPointer m_FixedMaskContainer{ nullptr };
  DataObjectContainerPointer m_MovingMaskContainer{ nullptr };
  DataObjectContainerPointer m_ResultImageContainer{ nullptr };

  /** A transform that is the result of registration. */
  ObjectPointer m_FinalTransform{ nullptr };

  /** The initial transform. */
  ObjectPointer m_InitialTransform{ nullptr };

  /** Transformation parameters map containing parameters that is the
   *  result of registration.
   */
  ParameterMapType m_TransformParametersMap{};

  FlatDirectionCosinesType m_OriginalFixedImageDirection{};

  /** InitDBIndex sets m_DBIndex by asking the ImageTypes
   * from the Configuration object and obtaining the corresponding
   * DB index from the ComponentDatabase.
   */
  int
  InitDBIndex() override;

  /** Helper function to obtain information from images on disk. */
  void
  GetImageInformationFromFile(const std::string & filename, ImageDimensionType & imageDimension) const;
};

} // end namespace elastix

#endif // end #ifndef elxElastixMain_h
