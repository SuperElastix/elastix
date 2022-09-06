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
#ifndef elxElastixTemplate_h
#define elxElastixTemplate_h

#include "elxElastixBase.h"
#include "elxFixedImagePyramidBase.h"
#include "elxImageSamplerBase.h"
#include "elxInterpolatorBase.h"
#include "elxMetricBase.h"
#include "elxMovingImagePyramidBase.h"
#include "elxOptimizerBase.h"
#include "elxRegistrationBase.h"
#include "elxResampleInterpolatorBase.h"
#include "elxResamplerBase.h"
#include "elxTransformBase.h"

// ITK header files:
#include <itkCommand.h>
#include <itkImage.h>
#include <itkObject.h>

#include <sstream>

/**
 * Macro that defines to functions. In the case of
 *   _name = Metric and _elxBaseType = MetricBaseType
 * this results in:
 * MetricBaseType * GetElxMetricBase() const;
 * MetricBaseType * GetElxMetricBase(unsigned int idx) const;
 *
 * The first function simply calls the second with argument = 0.
 * The second retrieves the metric component from the MetricContainer
 * and casts it to a MetricBaseType*;
 *
 * This macro is \#undef'ed at the end of this header file.
 */

#define elxGetBaseMacro(_name, _elxbasetype)                                                                           \
  _elxbasetype * GetElx##_name##Base(const unsigned int idx = 0) const                                                 \
  {                                                                                                                    \
    if (idx < this->GetNumberOf##_name##s())                                                                           \
    {                                                                                                                  \
      return dynamic_cast<_elxbasetype *>(this->Get##_name##Container()->ElementAt(idx).GetPointer());                 \
    }                                                                                                                  \
    return nullptr;                                                                                                    \
  }
// end elxGetBaseMacro

namespace elastix
{

/**
 * \class ElastixTemplate
 * \brief The main elastix class, which connects components
 * and invokes the BeforeRegistration(), BeforeEachResolution(),
 * etc. methods.
 *
 * The ElastixTemplate class ...
 *
 * \parameter WriteTransformParametersEachIteration: Controls whether
 *    to save a transform parameter file to disk in every iteration.\n
 *    example: <tt>(WriteTransformParametersEachIteration "true")</tt>\n
 *    This parameter can not be specified for each resolution separately.
 *    Default value: "false".
 * \parameter WriteTransformParametersEachResolution: Controls whether
 *    to save a transform parameter file to disk in every resolution.\n
 *    example: <tt>(WriteTransformParametersEachResolution "true")</tt>\n
 *    This parameter can not be specified for each resolution separately.
 *    Default value: "false".
 * \parameter UseDirectionCosines: Controls whether to use or ignore the
 * direction cosines (world matrix, transform matrix) set in the images.
 * Voxel spacing and image origin are always taken into account, regardless
 * the setting of this parameter.\n
 *    example: <tt>(UseDirectionCosines "true")</tt>\n
 * Default: true. Recommended: true. This parameter was introduced in
 * elastix 4.3, with a default value of false for backward compabitility.
 *  From elastix 4.8 the default value has been changed to true. Setting it
 *  to false means that you choose to ignore important information from the
 *  image, which relates voxel coordinates to world coordinates. Ignoring it
 *  may easily lead to left/right swaps for example, which could skrew up a
 *  (medical) analysis.
 *
 * \ingroup Kernel
 */

template <class TFixedImage, class TMovingImage>
class ITK_TEMPLATE_EXPORT ElastixTemplate final : public ElastixBase
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ElastixTemplate);

  /** Standard itk. */
  using Self = ElastixTemplate;
  using Superclass = ElastixBase;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ElastixTemplate, ElastixBase);

  /** Typedef's for this class. */
  using FixedImageType = TFixedImage;
  using MovingImageType = TMovingImage;
  using FixedImagePointer = typename FixedImageType::Pointer;
  using MovingImagePointer = typename MovingImageType::Pointer;

  /** For using the Dimensions. */
  itkStaticConstMacro(Dimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(FixedDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingDimension, unsigned int, MovingImageType::ImageDimension);

  /** Types for the masks. */
  using MaskPixelType = unsigned char;
  using FixedMaskType = itk::Image<MaskPixelType, FixedDimension>;
  using MovingMaskType = itk::Image<MaskPixelType, MovingDimension>;
  using FixedMaskPointer = typename FixedMaskType::Pointer;
  using MovingMaskPointer = typename MovingMaskType::Pointer;

  /** Typedef for the UseDirectionCosines option. */
  using FixedImageDirectionType = typename FixedImageType::DirectionType;

  /** BaseComponent. */
  using BaseComponentType = BaseComponent;

  /** A Pointer to a member function of a BaseComponentType. */
  using PtrToMemberFunction = void (BaseComponentType::*)();
  using PtrToMemberFunction2 = int (BaseComponentType::*)();

  /** Commands that react on Events and call Self::Function(). */
  using BeforeEachResolutionCommandType = itk::SimpleMemberCommand<Self>;
  using AfterEachResolutionCommandType = itk::SimpleMemberCommand<Self>;
  using AfterEachIterationCommandType = itk::SimpleMemberCommand<Self>;
  using BeforeEachResolutionCommandPointer = typename BeforeEachResolutionCommandType::Pointer;
  using AfterEachResolutionCommandPointer = typename AfterEachResolutionCommandType::Pointer;
  using AfterEachIterationCommandPointer = typename AfterEachIterationCommandType::Pointer;

  /** The elastix basecomponent types. */
  using FixedImagePyramidBaseType = FixedImagePyramidBase<Self>;
  using MovingImagePyramidBaseType = MovingImagePyramidBase<Self>;
  using InterpolatorBaseType = InterpolatorBase<Self>;
  using ImageSamplerBaseType = elx::ImageSamplerBase<Self>;
  using MetricBaseType = MetricBase<Self>;
  using OptimizerBaseType = OptimizerBase<Self>;
  using RegistrationBaseType = RegistrationBase<Self>;
  using ResamplerBaseType = ResamplerBase<Self>;
  using ResampleInterpolatorBaseType = ResampleInterpolatorBase<Self>;
  using TransformBaseType = elx::TransformBase<Self>;

  /** Typedef's for ApplyTransform.
   * \todo How useful is this? It is not consequently supported, since the
   * the input image is stored in the MovingImageContainer anyway.
   */
  using InputImageType = MovingImageType;
  using OutputImageType = MovingImageType;

  /** Typedef that is used in the elastix dll version. */
  using ParameterMapType = itk::ParameterMapInterface::ParameterMapType;

  /** Functions to set/get pointers to the elastix components.
   * Get the components as pointers to elxBaseType.
   */
  elxGetBaseMacro(FixedImagePyramid, FixedImagePyramidBaseType);
  elxGetBaseMacro(MovingImagePyramid, MovingImagePyramidBaseType);
  elxGetBaseMacro(Interpolator, InterpolatorBaseType);
  elxGetBaseMacro(ImageSampler, ImageSamplerBaseType);
  elxGetBaseMacro(Metric, MetricBaseType);
  elxGetBaseMacro(Optimizer, OptimizerBaseType);
  elxGetBaseMacro(Registration, RegistrationBaseType);
  elxGetBaseMacro(Resampler, ResamplerBaseType);
  elxGetBaseMacro(ResampleInterpolator, ResampleInterpolatorBaseType);
  elxGetBaseMacro(Transform, TransformBaseType);

  /** Get pointers to the images. They are obtained from the
   * {Fixed,Moving}ImageContainer and casted to the appropriate type.
   */

  FixedImageType *
  GetFixedImage(unsigned int idx = 0) const;

  MovingImageType *
  GetMovingImage(unsigned int idx = 0) const;

  /** Get pointers to the masks. They are obtained from the
   * {Fixed,Moving}MaskContainer and casted to the appropriate type.
   */

  FixedMaskType *
  GetFixedMask(unsigned int idx = 0) const;

  MovingMaskType *
  GetMovingMask(unsigned int idx = 0) const;

  /** Main functions:
   * Run() for registration, and ApplyTransform() for just
   * applying a transform to an image.
   */
  int
  Run() override;

  int
  ApplyTransform() override;

  /** The Callback functions. */
  int
  BeforeAll() override;

  int
  BeforeAllTransformix();

  void
  BeforeRegistration() override;

  void
  BeforeEachResolution() override;

  void
  AfterEachResolution() override;

  void
  AfterEachIteration() override;

  void
  AfterRegistration() override;

  /** Get the iteration number. */
  itkGetConstMacro(IterationCounter, unsigned int);

  /** Get the name of the current transform parameter file. */
  itkGetStringMacro(CurrentTransformParameterFileName);

  /** Get the original direction cosines of the fixed image. Returns
   * false if it failed to determine the original fixed image direction. In
   * that case the direction var is left unchanged. If no fixed image is
   * present, it tries to read it from the parameter file. */
  bool
  GetOriginalFixedImageDirection(FixedImageDirectionType & direction) const;

private:
  ElastixTemplate() = default;
  ~ElastixTemplate() override = default;

  /** CallBack commands. */
  BeforeEachResolutionCommandPointer m_BeforeEachResolutionCommand{};
  AfterEachIterationCommandPointer   m_AfterEachIterationCommand{};
  AfterEachResolutionCommandPointer  m_AfterEachResolutionCommand{};

  /** CreateTransformParameterFile. */
  void
  CreateTransformParameterFile(const std::string & FileName, const bool ToLog);

  /** CreateTransformParametersMap. */
  void
  CreateTransformParametersMap() override;

  /** Open the IterationInfoFile, where the table with iteration info is written to. */
  void
  OpenIterationInfoFile();

  /** Used by the callback functions, BeforeEachResolution() etc.).
   * This method calls a function in each component, in the following order:
   * \li Registration
   * \li Transform
   * \li ImageSampler
   * \li Metric
   * \li Interpolator
   * \li Optimizer
   * \li FixedImagePyramid
   * \li MovingImagePyramid
   * \li ResampleInterpolator
   * \li Resampler
   */
  void
  CallInEachComponent(PtrToMemberFunction func);

  int
  CallInEachComponentInt(PtrToMemberFunction2 func);

  /** Call in each component SetElastix(This) and set its ComponentLabel
   * (for example "Metric1"). This makes sure that the component knows its
   * own function in the registration process.
   */
  void
  ConfigureComponents(Self * This);

  /** Set the direction in the superclass' m_OriginalFixedImageDirection variable */
  void
  SetOriginalFixedImageDirection(const FixedImageDirectionType & arg);
};

} // end namespace elastix

#undef elxGetBaseMacro

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxElastixTemplate.hxx"
#endif

#endif // end #ifndef elxElastixTemplate_h
