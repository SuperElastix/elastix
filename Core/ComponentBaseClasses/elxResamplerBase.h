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
#include "elxPixelTypeToString.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
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
 * \note When WriteResultImage is false, the executable will not write a
 * result image, and the elastix library interface produces an empty image.
 *
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
 *    Choose from (unsigned) char, (unsigned) short, (unsigned) int, float, double, etc. Or choose a fixed width type:
 *    "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", or "float64". The width
 *    (number of bits) of a fixed width type is guaranteed to be the same on all supported platforms. \n example:
 * <tt>(ResultImagePixelType "unsigned short")</tt> \n
 *    The default is "short".
 * \parameter CompressResultImage: parameter to set if (lossless) compression
 *    of the written image is desired.\n
 *    example: <tt>(CompressResultImage "true")</tt> \n
 *    The default is "false".
 *
 * \ingroup Resamplers
 * \ingroup ComponentBaseClasses
 */

template <typename TElastix>
class ITK_TEMPLATE_EXPORT ResamplerBase : public BaseComponentSE<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ResamplerBase);

  /** Standard ITK stuff. */
  using Self = ResamplerBase;
  using Superclass = BaseComponentSE<TElastix>;

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ResamplerBase);

  /** Typedef's from superclass. */
  using typename Superclass::ElastixType;
  using typename Superclass::RegistrationType;

  /** Typedef's from elastix.
   * NB: it is assumed that fixed and moving image dimension are equal!  */
  using InputImageType = typename ElastixType::MovingImageType;
  using OutputImageType = typename ElastixType::MovingImageType;
  // typedef typename ElastixType::FixedImageType      OutputImageType;
  using CoordinateType = ElastixBase::CoordinateType;

  /** Other typedef's. */
  using ITKBaseType = itk::ResampleImageFilter<InputImageType, OutputImageType, CoordinateType>;

  /** Typedef's from ResampleImageFiler. */
  using TransformType = typename ITKBaseType::TransformType;
  using InterpolatorType = typename ITKBaseType::InterpolatorType;
  using SizeType = typename ITKBaseType::SizeType;
  using IndexType = typename ITKBaseType::IndexType;
  using SpacingType = typename ITKBaseType::SpacingType;
  using DirectionType = typename ITKBaseType::DirectionType;
  using OriginPointType = typename ITKBaseType::OriginPointType;
  using OutputPixelType = typename ITKBaseType::PixelType;

  /** Typedef that is used in the elastix dll version. */
  using ParameterMapType = typename ElastixType::ParameterMapType;

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, OutputImageType::ImageDimension);

  /** Retrieves this object as ITKBaseType. */
  ITKBaseType *
  GetAsITKBaseType()
  {
    return &(this->GetSelf());
  }


  /** Retrieves this object as ITKBaseType, to use in const functions. */
  const ITKBaseType *
  GetAsITKBaseType() const
  {
    return &(this->GetSelf());
  }


  /** Execute stuff before the actual transformation:
   * \li nothing here
   */
  virtual int
  BeforeAllTransformix()
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
  BeforeRegistrationBase() override;

  /** Execute stuff after each resolution:
   * \li Write the resulting output image.
   */
  void
  AfterEachResolutionBase() override;

  /** Execute stuff after each iteration:
   * \li Write the resulting output image.
   */
  void
  AfterEachIterationBase() override;

  /** Execute stuff after the registration:
   * \li Write the resulting output image.
   */
  void
  AfterRegistrationBase() override;

  /** Function to read transform-parameters from a file. */
  virtual void
  ReadFromFile();

  /** Function to write transform-parameters to a file. */
  void
  WriteToFile(std::ostream & transformationParameterInfo) const;

  /** Function to create transform-parameters map. */
  void
  CreateTransformParameterMap(ParameterMapType & parameterMap) const;

  /** Function to perform resample and write the result output image to a file. */
  void
  ResampleAndWriteResultImage(const std::string & filename, const bool showProgress);

  /** Function to create the result image in the format of an itk::Image. */
  virtual void
  CreateItkResultImage();

protected:
  /** The constructor. */
  ResamplerBase() = default;
  /** The destructor. */
  ~ResamplerBase() override = default;

  /** Method that sets the transform, the interpolator and the inputImage. */
  virtual void
  SetComponents();

private:
  elxDeclarePureVirtualGetSelfMacro(ITKBaseType);

  virtual ParameterMapType
  CreateDerivedTransformParameterMap() const
  {
    return {};
  }

  /** Function to write the result output image to a file. */
  void
  WriteResultImage(OutputImageType * imageimage, const std::string & filename, const bool showProgress);

  /** Release memory. */
  void
  ReleaseMemory();

  /** Casts the specified input image to the image type with the pixel type specified by the template argument. */
  template <typename TResultPixel>
  itk::SmartPointer<itk::ImageBase<ImageDimension>>
  CastImage(const InputImageType & inputImage) const
  {
    const auto castFilter =
      itk::CastImageFilter<InputImageType, itk::Image<TResultPixel, InputImageType::ImageDimension>>::New();
    castFilter->SetInput(&inputImage);
    castFilter->Update();
    return castFilter->GetOutput();
  }

  /** Casts the specified input image to the image type with the specified pixel type, if the function argument
   * `resultImagePixelType` matches the template argument `TResultPixel`. The function argument `resultImagePixelType`
   * should be of the fixed width form: "intN", "uintN", or "floatN", with 'N' specifying the number of bits. */
  template <typename TResultPixel>
  bool
  CastImageIfPixelTypesMatch(const std::string_view                              resultImagePixelType,
                             const InputImageType &                              inputImage,
                             itk::SmartPointer<itk::ImageBase<ImageDimension>> & outputImage) const
  {
    if (resultImagePixelType == PixelTypeToFixedWidthString<TResultPixel>())
    {
      outputImage = CastImage<TResultPixel>(inputImage);
      return true;
    }
    return false;
  }

  /** Casts the specified input image to the image type with the specified fixed width pixel type. */
  template <typename... TResultPixel>
  itk::SmartPointer<itk::ImageBase<ImageDimension>>
  CastImageAsSpecifiedByFixedWidthPixelType(const std::string_view resultImagePixelType,
                                            const InputImageType & inputImage) const
  {
    if (itk::SmartPointer<itk::ImageBase<ImageDimension>> outputImage;
        (CastImageIfPixelTypesMatch<TResultPixel>(resultImagePixelType, inputImage, outputImage) || ...))
    {
      return outputImage;
    }
    return nullptr;
  }
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxResamplerBase.hxx"
#endif

#endif // end #ifndef elxResamplerBase_h
