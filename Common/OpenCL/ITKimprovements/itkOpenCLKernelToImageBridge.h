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
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//

#ifndef itkOpenCLKernelToImageBridge_h
#define itkOpenCLKernelToImageBridge_h

#include "itkGPUImage.h"
#include "itkGPUDataManager.h"

#include "itkOpenCLKernel.h"

namespace itk
{
/** \class OpenCLKernelToImageBridge
 * \brief
 *
 * \ingroup OpenCL
 */
template <typename TImage>
class ITK_TEMPLATE_EXPORT OpenCLKernelToImageBridge
{
public:
  /** Standard class typedefs. */
  using Self = OpenCLKernelToImageBridge;

  /** Image class typedefs. */
  using ImageType = TImage;
  using ImagePointer = typename ImageType::ConstPointer;
  using ImageRegionType = typename ImageType::RegionType;
  using ImagePixelType = typename ImageType::PixelType;

  /** ImageDimension constants */
  itkStaticConstMacro(ImageDimension, unsigned int, TImage::ImageDimension);

  /** Run-time type information (and related methods). */
  virtual const char *
  GetNameOfClass() const
  {
    return "OpenCLKernelToImageBridge";
  }

  static void
  SetImageDataManager(OpenCLKernel &                         kernel,
                      const cl_uint                          argumentIndex,
                      const typename GPUDataManager::Pointer imageDataManager,
                      const bool                             updateCPU);

  static void
  SetImage(OpenCLKernel &                      kernel,
           const cl_uint                       argumentIndex,
           const typename ImageType::Pointer & image,
           const bool                          updateCPU);

  static void
  SetImageMetaData(OpenCLKernel &                      kernel,
                   const cl_uint                       argumentIndex,
                   const typename ImageType::Pointer & image,
                   typename GPUDataManager::Pointer &  imageMetaDataManager);

  static void
  SetDirection(OpenCLKernel & kernel, const cl_uint argumentIndex, const typename ImageType::DirectionType & direction);

  static void
  SetSize(OpenCLKernel & kernel, const cl_uint argumentIndex, const typename ImageType::SizeType & size);

  static void
  SetOrigin(OpenCLKernel & kernel, const cl_uint argumentIndex, const typename ImageType::PointType & origin);

protected:
  OpenCLKernelToImageBridge();
  virtual ~OpenCLKernelToImageBridge() {}

private:
  OpenCLKernelToImageBridge(const Self & other); // purposely not
                                                 // implemented
  const Self &
  operator=(const Self &); // purposely not

  // implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkOpenCLKernelToImageBridge.hxx"
#endif

#endif /* itkOpenCLKernelToImageBridge_h */
