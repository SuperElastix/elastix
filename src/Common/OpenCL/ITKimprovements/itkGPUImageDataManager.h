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
/*=========================================================================
*
*  Copyright Insight Software Consortium
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
#ifndef __itkGPUImageDataManager_h
#define __itkGPUImageDataManager_h

#include <itkObject.h>
#include <itkTimeStamp.h>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkGPUImage.h"
#include "itkGPUDataManager.h"
#include "itkSimpleFastMutexLock.h"

namespace itk
{
/**
 * \class GPUImage Data Management
 *
 * DataManager for GPUImage. This class will take care of data synchronization
 * between CPU Image and GPU Image.
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 * \ingroup ITKGPUCommon
 */
template< typename TPixel, unsigned int NDimension >
class GPUImage;

template< typename ImageType >
class ITKOpenCL_EXPORT GPUImageDataManager : public GPUDataManager
{
  // allow GPUKernelManager to access GPU buffer pointer
  friend class OpenCLKernelManager;
  friend class GPUImage< typename ImageType::PixelType, ImageType::ImageDimension >;

public:

  typedef GPUImageDataManager        Self;
  typedef GPUDataManager             Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  itkNewMacro( Self );
  itkTypeMacro( GPUImageDataManager, GPUDataManager );

  void SetImagePointer( typename ImageType::Pointer img );

  /** actual GPU->CPU memory copy takes place here */
  virtual void UpdateCPUBuffer();

  /** actual CPU->GPU memory copy takes place here */
  virtual void UpdateGPUBuffer();

  /** Grafting GPU Image Data */
  virtual void Graft( const GPUImageDataManager * data );

protected:

  GPUImageDataManager() { m_Image = NULL; }
  virtual ~GPUImageDataManager() {}

private:

  GPUImageDataManager( const Self & );   // purposely not implemented
  void operator=( const Self & );        // purposely not implemented

  typename ImageType::Pointer m_Image;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUImageDataManager.hxx"
#endif

#endif
