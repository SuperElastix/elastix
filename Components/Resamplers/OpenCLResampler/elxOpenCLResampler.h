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
#ifndef elxOpenCLResampler_h
#define elxOpenCLResampler_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "elxOpenCLSupportedImageTypes.h"

#include "itkGPUResampleImageFilter.h"
#include "itkGPUAdvancedCombinationTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

namespace elastix
{

/**
 * \class OpenCLResampler
 * \brief A resampler based on the itk::GPUResampleImageFilter.
 * The parameters used in this class are:
 * \parameter Resampler: Select this resampler as follows:\n
 *    <tt>(Resampler "OpenCLResampler")</tt>
 * \parameter Resampler: Enable the OpenCL resampler as follows:\n
 *    <tt>(OpenCLResamplerUseOpenCL "true")</tt>
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \ingroup Resamplers
 */

template <class TElastix>
class OpenCLResampler
  : public itk::ResampleImageFilter<typename ResamplerBase<TElastix>::InputImageType,
                                    typename ResamplerBase<TElastix>::OutputImageType,
                                    typename ResamplerBase<TElastix>::CoordRepType>
  , public ResamplerBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OpenCLResampler);

  /** Standard ITK-stuff. */
  using Self = OpenCLResampler;

  using Superclass1 = itk::ResampleImageFilter<typename ResamplerBase<TElastix>::InputImageType,
                                               typename ResamplerBase<TElastix>::OutputImageType,
                                               typename ResamplerBase<TElastix>::CoordRepType>;
  using Superclass2 = ResamplerBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCLResampler, ResampleImageFilter);

  /** Name of this class.
   * Use this name in the parameter file to select this specific resampler. \n
   * example: <tt>(Resampler "OpenCLResampler")</tt>\n
   */
  elxClassNameMacro("OpenCLResampler");

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::TransformType;

  using typename Superclass1::InputImageType;
  using InputImagePixelType = typename InputImageType::PixelType;

  using typename Superclass1::OutputImageType;
  using OutputImagePixelType = typename OutputImageType::PixelType;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** GPU Typedefs for GPU image and GPU resampler. */
  using GPUInputImageType = itk::GPUImage<InputImagePixelType, InputImageType::ImageDimension>;
  using GPUInputImagePointer = typename GPUInputImageType::Pointer;
  using GPUOutputImageType = itk::GPUImage<OutputImagePixelType, OutputImageType::ImageDimension>;
  using GPUInterpolatorPrecisionType = float;

  using GPUResamplerType =
    itk::GPUResampleImageFilter<GPUInputImageType, GPUOutputImageType, GPUInterpolatorPrecisionType>;
  using GPUResamplerPointer = typename GPUResamplerType::Pointer;

  using typename Superclass2::ParameterMapType;

  /** Set the transform. */
  void
  SetTransform(const TransformType * _arg) override;

  /** Set the interpolator. */
  void
  SetInterpolator(InterpolatorType * _arg) override;

  /** Do some things before registration. */
  void
  BeforeRegistration() override;

  /** Function to read parameters from a file. */
  void
  ReadFromFile() override;

protected:
  /** The constructor. */
  OpenCLResampler();
  /** The destructor. */
  ~OpenCLResampler() override = default;

  /** This method performs all configuration for GPU resampler. */
  void
  BeforeGenerateData();

  /** Executes GPU resampler. */
  void
  GenerateData() override;

  /** Transform copier */
  using InterpolatorPrecisionType = typename ResamplerBase<TElastix>::CoordRepType;
  using AdvancedCombinationTransformType =
    typename itk::AdvancedCombinationTransform<InterpolatorPrecisionType, OutputImageType::ImageDimension>;
  using TransformCopierType = typename itk::GPUAdvancedCombinationTransformCopier<OpenCLImageTypes,
                                                                                  OpenCLImageDimentions,
                                                                                  AdvancedCombinationTransformType,
                                                                                  float>;
  using TransformCopierPointer = typename TransformCopierType::Pointer;
  using GPUTransformPointer = typename TransformCopierType::GPUComboTransformPointer;

  /** Interpolator copier */
  using InterpolatorInputImageType = typename InterpolatorType::InputImageType;
  using InterpolatorCoordRepType = typename InterpolatorType::CoordRepType;
  using InterpolateImageFunctionType =
    itk::InterpolateImageFunction<InterpolatorInputImageType, InterpolatorCoordRepType>;
  using InterpolateCopierType =
    typename itk::GPUInterpolatorCopier<OpenCLImageTypes, OpenCLImageDimentions, InterpolateImageFunctionType, float>;
  using InterpolateCopierPointer = typename InterpolateCopierType::Pointer;
  using GPUExplicitInterpolatorPointer = typename InterpolateCopierType::GPUExplicitInterpolatorPointer;

private:
  elxOverrideGetSelfMacro;

  /** Creates a map of the parameters specific for this (derived) resampler type. */
  ParameterMapType
  CreateDerivedTransformParametersMap() const override;

  /** Helper method to report switching to CPU mode. */
  void
  SwitchingToCPUAndReport(const bool configError);

  /** Helper method to report to elastix log. */
  void
  ReportToLog();

  TransformCopierPointer   m_TransformCopier;
  InterpolateCopierPointer m_InterpolatorCopier;
  GPUResamplerPointer      m_GPUResampler;
  bool                     m_GPUResamplerReady;
  bool                     m_GPUResamplerCreated;
  bool                     m_ContextCreated;
  bool                     m_UseOpenCL;
};

// end class OpenCLResampler

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxOpenCLResampler.hxx"
#endif

#endif // end #ifndef elxOpenCLResampler_h
