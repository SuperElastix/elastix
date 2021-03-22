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
#ifndef elxResamplerBase_h
#define elxResamplerBase_h

/** Needed for the macros */
#include "elxMacro.h"

#include "elxBaseComponentSE.h"
#include "itkResampleImageFilter.h"
#include "elxProgressCommand.h"

namespace elastix
{
/**
 * \class ResampleBase
 * \brief This class is the elastix base class for all Resamplers.
 *
 * This class contains all the common functionality for Resamplers.
 *
 * The parameters used in this class are:
 * \parameter WriteResultImage: flag to determine if the final
 *    result image is resampled and written. Choose from {"true", "false"} \n
 *    example: <tt>(WriteResultImage "false")</tt> \n
 *    The default is "true".
 * \parameter WriteResultImageAfterEachResolution: flag to determine if the intermediate
 *    result image is resampled and written after each resolution. Choose from {"true", "false"} \n
 *    example: <tt>(WriteResultImageAfterEachResolution "true" "false" "true")</tt> \n
 *    The default is "false" for each resolution.
 * \parameter WriteResultImageAfterEachIteration: flag to determine if the intermediate
 *    result image is resampled and written after each iteration. Choose from {"true", "false"} \n
 *    example: <tt>(WriteResultImageAfterEachIteration "true" "false" "true")</tt> \n
 *    The default is "false" for each iteration.\n
 *    Note that this option is only useful for debugging / tuning purposes.
 * \parameter ResultImageFormat: parameter to set the image file format to
 *    to which the resampled image is written to.\n
 *    example: <tt>(ResultImageFormat "mhd")</tt> \n
 *    The default is "mhd".
 * \parameter ResultImagePixelType: parameter to set the pixel type,
 *    used for resampling the moving image. If this is different from
 *    the input pixel type you are casting your data. This is done
 *    using standard c-style casts, so TAKE CARE that you are not
 *    throwing away data (for example when going from unsigned to signed,
 *    or from float to char).\n
 *    Choose from (unsigned) char, (unsigned) short, float, double, etc.\n
 *    example: <tt>(ResultImagePixelType "unsigned short")</tt> \n
 *    The default is "short".
 * \parameter CompressResultImage: parameter to set if (lossless) compression
 *    of the written image is desired.\n
 *    example: <tt>(CompressResultImage "true")</tt> \n
 *    The default is "false".
 *
 * \ingroup Resamplers
 * \ingroup ComponentBaseClasses
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT ResamplerBase : public BaseComponentSE<TElastix>
{
public:
  /** Standard ITK stuff. */
  typedef ResamplerBase             Self;
  typedef BaseComponentSE<TElastix> Superclass;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ResamplerBase, BaseComponentSE);

  /** Typedef's from superclass. */
  typedef typename Superclass::ElastixType          ElastixType;
  typedef typename Superclass::ElastixPointer       ElastixPointer;
  typedef typename Superclass::ConfigurationType    ConfigurationType;
  typedef typename Superclass::ConfigurationPointer ConfigurationPointer;
  typedef typename Superclass::RegistrationType     RegistrationType;
  typedef typename Superclass::RegistrationPointer  RegistrationPointer;

  /** Typedef's from elastix.
   * NB: it is assumed that fixed and moving image dimension are equal!  */
  typedef typename ElastixType::MovingImageType InputImageType;
  typedef typename ElastixType::MovingImageType OutputImageType;
  // typedef typename ElastixType::FixedImageType      OutputImageType;
  typedef ElastixBase::CoordRepType CoordRepType;

  /** Other typedef's. */
  typedef itk::ResampleImageFilter<InputImageType, OutputImageType, CoordRepType> ITKBaseType;

  /** Typedef's from ResampleImageFiler. */
  typedef typename ITKBaseType::TransformType    TransformType;
  typedef typename ITKBaseType::InterpolatorType InterpolatorType;
  typedef typename ITKBaseType::SizeType         SizeType;
  typedef typename ITKBaseType::IndexType        IndexType;
  typedef typename ITKBaseType::SpacingType      SpacingType;
  typedef typename ITKBaseType::DirectionType    DirectionType;
  typedef typename ITKBaseType::OriginPointType  OriginPointType;
  typedef typename ITKBaseType::PixelType        OutputPixelType;

  /** Typedef that is used in the elastix dll version. */
  typedef typename ElastixType::ParameterMapType ParameterMapType;

  /** Typedef for the ProgressCommand. */
  typedef elx::ProgressCommand ProgressCommandType;

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, OutputImageType::ImageDimension);

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType(void)
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType(void) const
  {
    return &(this->GetSelf());
  }


  /** Execute stuff before the actual transformation:
   * \li nothing here
   */
  virtual int
  BeforeAllTransformix(void)
  {
    return 0;
  }

  /** Execute stuff before the actual registration:
   * \li Set all components into the resampler, such as the transform
   *    interpolator, input.
   * \li Set output image information, such as size, spacing, etc.
   * \li Set the default pixel value.
   */
  void
  BeforeRegistrationBase(void) override;

  /** Execute stuff after each resolution:
   * \li Write the resulting output image.
   */
  void
  AfterEachResolutionBase(void) override;

  /** Execute stuff after each iteration:
   * \li Write the resulting output image.
   */
  void
  AfterEachIterationBase(void) override;

  /** Execute stuff after the registration:
   * \li Write the resulting output image.
   */
  void
  AfterRegistrationBase(void) override;

  /** Function to read transform-parameters from a file. */
  virtual void
  ReadFromFile(void);

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(xl::xoutsimple & transformationParameterInfo) const;

  /** Function to create transform-parameters map. */
  void
  CreateTransformParametersMap(ParameterMapType & parameterMap) const;

  /** Function to perform resample and write the result output image to a file. */
  virtual void
  ResampleAndWriteResultImage(const char * filename, const bool & showProgress = true);

  /** Function to write the result output image to a file. */
  virtual void
  WriteResultImage(OutputImageType * imageimage, const char * filename, const bool & showProgress = true);

  /** Function to create the result image in the format of an itk::Image. */
  virtual void
  CreateItkResultImage(void);

protected:
  /** The constructor. */
  ResamplerBase();
  /** The destructor. */
  ~ResamplerBase() override = default;

  /** Method that sets the transform, the interpolator and the inputImage. */
  virtual void
  SetComponents(void);

  /** Variable that defines to print the progress or not. */
  bool m_ShowProgress;

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  virtual ParameterMapType
  CreateDerivedTransformParametersMap() const
  {
    return {};
  }

  /** The deleted copy constructor. */
  ResamplerBase(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  /** Release memory. */
  void
  ReleaseMemory(void);
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxResamplerBase.hxx"
#endif

#endif // end #ifndef elxResamplerBase_h
