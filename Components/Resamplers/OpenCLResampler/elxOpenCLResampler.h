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
  /** Standard ITK-stuff. */
  typedef OpenCLResampler Self;

  typedef itk::ResampleImageFilter<typename ResamplerBase<TElastix>::InputImageType,
                                   typename ResamplerBase<TElastix>::OutputImageType,
                                   typename ResamplerBase<TElastix>::CoordRepType>
                                        Superclass1;
  typedef ResamplerBase<TElastix>       Superclass2;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

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
  typedef typename Superclass1::InterpolatorType InterpolatorType;
  typedef typename Superclass1::TransformType    TransformType;

  typedef typename Superclass1::InputImageType InputImageType;
  typedef typename InputImageType::PixelType   InputImagePixelType;

  typedef typename Superclass1::OutputImageType OutputImageType;
  typedef typename OutputImageType::PixelType   OutputImagePixelType;
  typedef typename OutputImageType::RegionType  OutputImageRegionType;

  /** GPU Typedefs for GPU image and GPU resampler. */
  typedef itk::GPUImage<InputImagePixelType, InputImageType::ImageDimension>   GPUInputImageType;
  typedef typename GPUInputImageType::Pointer                                  GPUInputImagePointer;
  typedef itk::GPUImage<OutputImagePixelType, OutputImageType::ImageDimension> GPUOutputImageType;
  typedef float                                                                GPUInterpolatorPrecisionType;

  typedef itk::GPUResampleImageFilter<GPUInputImageType, GPUOutputImageType, GPUInterpolatorPrecisionType>
                                             GPUResamplerType;
  typedef typename GPUResamplerType::Pointer GPUResamplerPointer;

  typedef typename Superclass2::ParameterMapType ParameterMapType;

  /** Set the transform. */
  virtual void
  SetTransform(const TransformType * _arg);

  /** Set the interpolator. */
  virtual void
  SetInterpolator(InterpolatorType * _arg);

  /** Do some things before registration. */
  virtual void
  BeforeRegistration(void);

  /** Function to read parameters from a file. */
  virtual void
  ReadFromFile(void);

protected:
  /** The constructor. */
  OpenCLResampler();
  /** The destructor. */
  virtual ~OpenCLResampler() = default;

  /** This method performs all configuration for GPU resampler. */
  void
  BeforeGenerateData(void);

  /** Executes GPU resampler. */
  virtual void
  GenerateData(void);

  /** Transform copier */
  typedef typename ResamplerBase<TElastix>::CoordRepType InterpolatorPrecisionType;
  typedef typename itk::AdvancedCombinationTransform<InterpolatorPrecisionType, OutputImageType::ImageDimension>
    AdvancedCombinationTransformType;
  typedef typename itk::GPUAdvancedCombinationTransformCopier<OpenCLImageTypes,
                                                              OpenCLImageDimentions,
                                                              AdvancedCombinationTransformType,
                                                              float>
                                                                 TransformCopierType;
  typedef typename TransformCopierType::Pointer                  TransformCopierPointer;
  typedef typename TransformCopierType::GPUComboTransformPointer GPUTransformPointer;

  /** Interpolator copier */
  typedef typename InterpolatorType::InputImageType InterpolatorInputImageType;
  typedef typename InterpolatorType::CoordRepType   InterpolatorCoordRepType;
  typedef itk::InterpolateImageFunction<InterpolatorInputImageType, InterpolatorCoordRepType>
    InterpolateImageFunctionType;
  typedef
    typename itk::GPUInterpolatorCopier<OpenCLImageTypes, OpenCLImageDimentions, InterpolateImageFunctionType, float>
                                                                         InterpolateCopierType;
  typedef typename InterpolateCopierType::Pointer                        InterpolateCopierPointer;
  typedef typename InterpolateCopierType::GPUExplicitInterpolatorPointer GPUExplicitInterpolatorPointer;

private:
  elxOverrideGetSelfMacro;
  elxOverrideGetSelfMacro;
  /** Creates a map of the parameters specific for this (derived) resampler type. */
  ParameterMapType
  CreateDerivedTransformParametersMap(void) const override;

  /** The deleted copy constructor. */
  OpenCLResampler(const Self &) = delete;
  /** The deleted assignment operator. */
  void
  operator=(const Self &) = delete;

  /** Helper method to report switching to CPU mode. */
  void
  SwitchingToCPUAndReport(const bool configError);

  /** Helper method to report to elastix log. */
  void
  ReportToLog(void);

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
