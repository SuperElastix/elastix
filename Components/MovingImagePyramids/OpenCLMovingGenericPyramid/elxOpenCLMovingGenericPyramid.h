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
#ifndef elxOpenCLMovingGenericPyramid_h
#define elxOpenCLMovingGenericPyramid_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "elxMovingGenericPyramid.h"
#include "itkGPUImage.h"

namespace elastix
{

/**
 * \class OpenCLMovingGenericPyramid
 * \brief A pyramid based on the itk::GenericMultiResolutionPyramidImageFilter.
 * The parameters used in this class are:
 * \parameter Pyramid: Select this pyramid as follows:\n
 *    <tt>(MovingImagePyramid "OpenCLMovingGenericImagePyramid")</tt>
 * \parameter Pyramid: Enable the OpenCL pyramid as follows:\n
 *    <tt>(OpenCLMovingGenericImagePyramidUseOpenCL "true")</tt>
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 * \sa MovingGenericPyramid
 * \ingroup ImagePyramids
 */

template <class TElastix>
class ITK_TEMPLATE_EXPORT OpenCLMovingGenericPyramid : public MovingGenericPyramid<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(OpenCLMovingGenericPyramid);

  /** Standard ITK-stuff. */
  using Self = OpenCLMovingGenericPyramid;
  using Superclass = MovingGenericPyramid<TElastix>;
  using Superclass1 = typename MovingGenericPyramid<TElastix>::Superclass1;
  using Superclass2 = typename MovingGenericPyramid<TElastix>::Superclass2;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCLMovingGenericPyramid, MovingGenericPyramid);

  /** Name of this class.
   * Use this name in the parameter file to select this specific pyramid. \n
   * example: <tt>(MovingImagePyramid "OpenCLMovingGenericImagePyramid")</tt>\n
   */
  elxClassNameMacro("OpenCLMovingGenericImagePyramid");

  /** Get the ImageDimension. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass1::ImageDimension);

  /** Typedefs inherited from the superclass. */
  using typename Superclass1::InputImageType;
  using typename Superclass1::OutputImageType;
  using InputImagePixelType = typename Superclass1::InputImageType::PixelType;
  using OutputImagePixelType = typename Superclass1::OutputImageType::PixelType;

  /** Typedefs for factory. */
  using ObjectFactoryBasePointer = typename itk::ObjectFactoryBase::Pointer;

  /** GPU Typedefs for GPU image and GPU filter. */
  using GPUInputImageType = itk::GPUImage<InputImagePixelType, InputImageType::ImageDimension>;
  using GPUInputImagePointer = typename GPUInputImageType::Pointer;
  using GPUOutputImageType = itk::GPUImage<OutputImagePixelType, OutputImageType::ImageDimension>;

  using GPUPyramidType = itk::GenericMultiResolutionPyramidImageFilter<GPUInputImageType, GPUOutputImageType, float>;
  using GPUPyramidPointer = typename GPUPyramidType::Pointer;

  /** Do some things before registration. */
  void
  BeforeRegistration() override;

  /** Function to read parameters from a file. */
  virtual void
  ReadFromFile();

protected:
  /** This method performs all configuration for GPU pyramid. */
  void
  BeforeGenerateData();

  /** Executes GPU pyramid. */
  void
  GenerateData() override;

  /** The constructor. */
  OpenCLMovingGenericPyramid();
  /** The destructor. */
  ~OpenCLMovingGenericPyramid() override = default;

private:
  elxOverrideGetSelfMacro;

  /** Register/Unregister factories. */
  void
  RegisterFactories();

  void
  UnregisterFactories();

  /** Helper method to report switching to CPU mode. */
  void
  SwitchingToCPUAndReport(const bool configError);

  /** Helper method to report to elastix log. */
  void
  ReportToLog();

  GPUPyramidPointer                     m_GPUPyramid;
  bool                                  m_GPUPyramidReady;
  bool                                  m_GPUPyramidCreated;
  bool                                  m_ContextCreated;
  bool                                  m_UseOpenCL;
  std::vector<ObjectFactoryBasePointer> m_Factories;
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxOpenCLMovingGenericPyramid.hxx"
#endif

#endif // end #ifndef elxOpenCLMovingGenericPyramid_h
