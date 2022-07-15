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

#ifndef itkGPUKernelManagerHelperFunctions_h
#define itkGPUKernelManagerHelperFunctions_h

#include "itkGPUImage.h"

#include "itkOpenCLKernelManager.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLOstreamSupport.h"
#include <string>

namespace itk
{
// Definition of GPUImageBase 1D
typedef struct
{
  cl_float Direction;
  cl_float IndexToPhysicalPoint;
  cl_float PhysicalPointToIndex;
  cl_float Spacing;
  cl_float Origin;
  cl_uint  Size;
} GPUImageBase1D;

// Definition of GPUImageBase 2D
typedef struct
{
  cl_float4 Direction;
  cl_float4 IndexToPhysicalPoint;
  cl_float4 PhysicalPointToIndex;
  cl_float2 Spacing;
  cl_float2 Origin;
  cl_uint2  Size;
} GPUImageBase2D;

// Definition of GPUImageBase 3D
typedef struct
{
  cl_float16 Direction;            // OpenCL does not have float9
  cl_float16 IndexToPhysicalPoint; // OpenCL does not have float9
  cl_float16 PhysicalPointToIndex; // OpenCL does not have float9
  cl_float3  Spacing;
  cl_float3  Origin;
  cl_uint3   Size;
} GPUImageBase3D;

//----------------------------------------------------------------------------
template <typename ImageType>
void
SetKernelWithDirection(const typename ImageType::DirectionType & dir,
                       cl_float &                                direction1d,
                       cl_float4 &                               direction2d,
                       cl_float16 &                              direction3d)
{
  const unsigned int ImageDim = (unsigned int)(ImageType::ImageDimension);

  if (ImageDim == 1)
  {
    float direction = 0.0f;
    direction = static_cast<float>(dir[0][0]);
    direction1d = direction;
  }
  else if (ImageDim == 2)
  {
    float        direction[4];
    unsigned int index = 0;
    for (unsigned int i = 0; i < ImageDim; ++i)
    {
      for (unsigned int j = 0; j < ImageDim; ++j)
      {
        direction[index] = static_cast<float>(dir[i][j]);
        ++index;
      }
    }
    for (unsigned int i = 0; i < 4; ++i)
    {
      direction2d.s[i] = direction[i];
    }
  }
  else
  {
    // OpenCL does not support float9 therefore we are using float16
    float        direction[16];
    unsigned int index = 0;
    for (unsigned int i = 0; i < ImageDim; ++i)
    {
      for (unsigned int j = 0; j < ImageDim; ++j)
      {
        direction[index] = static_cast<float>(dir[i][j]);
        ++index;
      }
    }
    for (unsigned int i = 9; i < 16; ++i)
    {
      direction[i] = 0.0f;
    }
    for (unsigned int i = 0; i < 16; ++i)
    {
      direction3d.s[i] = direction[i];
    }
  }
}


template <typename ImageType>
void
SetKernelWithITKImage(OpenCLKernelManager::Pointer &      kernelManager,
                      const int                           kernelIdx,
                      cl_uint &                           argIdx,
                      const typename ImageType::Pointer & image,
                      typename GPUDataManager::Pointer &  imageBase,
                      const bool                          copyImage,
                      const bool                          copyImageBase)
{
  if (ImageType::ImageDimension > 3 || ImageType::ImageDimension < 1)
  {
    itkGenericExceptionMacro("SetKernelWithITKImage only supports 1D/2D/3D images.");
  }
  // Perform the safe check
  if (kernelManager.IsNull())
  {
    itkGenericExceptionMacro(<< "The kernel manager is NULL.");
    return;
  }

  if (image.IsNull())
  {
    itkGenericExceptionMacro(<< "The ITK image is NULL. "
                                "Unable to set ITK image information to the kernel manager.");
    return;
  }

  // Set ITK image to the kernelManager
  if (copyImage)
  {
    kernelManager->SetKernelArgWithImage(kernelIdx, argIdx++, image->GetGPUDataManager());
  }

  // Set ITK image base to the kernelManager
  if (copyImageBase)
  {
    const unsigned int ImageDim = (unsigned int)(ImageType::ImageDimension);
    GPUImageBase1D     imageBase1D;
    GPUImageBase2D     imageBase2D;
    GPUImageBase3D     imageBase3D;

    // Set size
    typename ImageType::RegionType largestPossibleRegion;
    if (image.IsNotNull())
    {
      largestPossibleRegion = image->GetLargestPossibleRegion();
    }

    using size_type = unsigned int;
    size_type size[ImageType::ImageDimension];
    for (unsigned int i = 0; i < ImageDim; ++i)
    {
      if (image.IsNotNull())
      {
        size[i] = static_cast<size_type>(largestPossibleRegion.GetSize()[i]);
      }
      else
      {
        size[i] = 0;
      }
    }
    if (ImageDim == 1)
    {
      imageBase1D.Size = size[0];
    }
    else if (ImageDim == 2)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase2D.Size.s[i] = size[i];
      }
    }
    else if (ImageDim == 3)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase3D.Size.s[i] = size[i];
      }
    }

    // Set spacing
    float spacing[ImageType::ImageDimension];
    for (unsigned int i = 0; i < ImageDim; ++i)
    {
      if (image.IsNotNull())
      {
        spacing[i] = static_cast<float>(image->GetSpacing()[i]);
      }
      else
      {
        spacing[i] = 0.0f;
      }
    }
    if (ImageDim == 1)
    {
      imageBase1D.Spacing = spacing[0];
    }
    else if (ImageDim == 2)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase2D.Spacing.s[i] = spacing[i];
      }
    }
    else if (ImageDim == 3)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase3D.Spacing.s[i] = spacing[i];
      }
    }

    // Set origin
    float origin[ImageType::ImageDimension];
    for (unsigned int i = 0; i < ImageDim; ++i)
    {
      if (image.IsNotNull())
      {
        origin[i] = static_cast<float>(image->GetOrigin()[i]);
      }
      else
      {
        origin[i] = 0.0f;
      }
    }
    if (ImageDim == 1)
    {
      imageBase1D.Origin = origin[0];
    }
    else if (ImageDim == 2)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase2D.Origin.s[i] = origin[i];
      }
    }
    else if (ImageDim == 3)
    {
      for (unsigned int i = 0; i < ImageDim; ++i)
      {
        imageBase3D.Origin.s[i] = origin[i];
      }
    }

    if (image.IsNotNull())
    {
      SetKernelWithDirection<ImageType>(
        image->GetDirection(), imageBase1D.Direction, imageBase2D.Direction, imageBase3D.Direction);

      SetKernelWithDirection<ImageType>(image->GetIndexToPhysicalPoint(),
                                        imageBase1D.IndexToPhysicalPoint,
                                        imageBase2D.IndexToPhysicalPoint,
                                        imageBase3D.IndexToPhysicalPoint);

      SetKernelWithDirection<ImageType>(image->GetPhysicalPointToIndex(),
                                        imageBase1D.PhysicalPointToIndex,
                                        imageBase2D.PhysicalPointToIndex,
                                        imageBase3D.PhysicalPointToIndex);
    }
    else
    {
      typename ImageType::DirectionType dir_null;
      dir_null.Fill(0);
      SetKernelWithDirection<ImageType>(dir_null, imageBase1D.Direction, imageBase2D.Direction, imageBase3D.Direction);

      SetKernelWithDirection<ImageType>(
        dir_null, imageBase1D.IndexToPhysicalPoint, imageBase2D.IndexToPhysicalPoint, imageBase3D.IndexToPhysicalPoint);

      SetKernelWithDirection<ImageType>(
        dir_null, imageBase1D.PhysicalPointToIndex, imageBase2D.PhysicalPointToIndex, imageBase3D.PhysicalPointToIndex);
    }

    // Set image base
    imageBase->Initialize();
    imageBase->SetBufferFlag(CL_MEM_READ_ONLY);
    if (ImageDim == 1)
    {
      imageBase->SetBufferSize(sizeof(GPUImageBase1D));
    }
    else if (ImageDim == 2)
    {
      imageBase->SetBufferSize(sizeof(GPUImageBase2D));
    }
    else if (ImageDim == 3)
    {
      imageBase->SetBufferSize(sizeof(GPUImageBase3D));
    }

    imageBase->Allocate();

    if (ImageDim == 1)
    {
      imageBase->SetCPUBufferPointer(&imageBase1D);
    }
    else if (ImageDim == 2)
    {
      imageBase->SetCPUBufferPointer(&imageBase2D);
    }
    else if (ImageDim == 3)
    {
      imageBase->SetCPUBufferPointer(&imageBase3D);
    }

    imageBase->SetGPUDirtyFlag(true);
    imageBase->UpdateGPUBuffer();

    kernelManager->SetKernelArgWithImage(kernelIdx, argIdx++, imageBase);
  }
}


} // end namespace itk

#endif /* itkGPUKernelManagerHelperFunctions_h */
