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
#ifndef itkOpenCLKernelToImageBridge_hxx
#define itkOpenCLKernelToImageBridge_hxx

#include "itkOpenCLKernelToImageBridge.h"
#include "itkOpenCLMacro.h"

// begin of ITKOpenCLKernelToImageBridge namespace
namespace ITKOpenCLKernelToImageBridge
{
// Definition of OCLImageBase 1D
typedef struct
{
  cl_float Direction;
  cl_float IndexToPhysicalPoint;
  cl_float PhysicalPointToIndex;
  cl_float Spacing;
  cl_float Origin;
  cl_uint  Size;
} OCLImageBase1D;

// Definition of OCLImageBase 2D
typedef struct
{
  cl_float4 Direction;
  cl_float4 IndexToPhysicalPoint;
  cl_float4 PhysicalPointToIndex;
  cl_float2 Spacing;
  cl_float2 Origin;
  cl_uint2  Size;
} OCLImageBase2D;

// Definition of OCLImageBase 3D
typedef struct
{
  cl_float16 Direction;            // OpenCL does not have float9
  cl_float16 IndexToPhysicalPoint; // OpenCL does not have float9
  cl_float16 PhysicalPointToIndex; // OpenCL does not have float9
  cl_float3  Spacing;
  cl_float3  Origin;
  cl_uint3   Size;
} OCLImageBase3D;
} // namespace ITKOpenCLKernelToImageBridge

namespace itk
{
//------------------------------------------------------------------------------
template <typename TImage>
void
OpenCLKernelToImageBridge<TImage>::SetImageDataManager(OpenCLKernel &                         kernel,
                                                       const cl_uint                          argumentIndex,
                                                       const typename GPUDataManager::Pointer imageDataManager,
                                                       const bool                             updateCPU)
{
  if (ImageType::ImageDimension > 3 || ImageType::ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetImageDataManager"
                             " supports only 1D/2D/3D images.");
    return;
  }

  cl_int error;

  if (imageDataManager.IsNotNull() && imageDataManager->GetBufferSize() > 0)
  {
    error = kernel.SetArg(argumentIndex, imageDataManager->GetGPUBufferPointer(), sizeof(cl_mem));
  }
  else
  {
    // According OpenCL specification clSetKernelArg arg_value
    // could be NULL object.
    cl_mem null_buffer = nullptr;
    error = kernel.SetArg(argumentIndex, &null_buffer, sizeof(cl_mem));
  }

  if (error != CL_SUCCESS)
  {
    itkOpenCLWarningMacroGeneric(
      << "Setting kernel argument failed with OpenCLKernelToImageBridge::SetImage(kernel name: '" << kernel.GetName()
      << "', " << argumentIndex << ")");
  }

  if (updateCPU)
  {
    imageDataManager->SetCPUBufferDirty();
  }

  kernel.GetContext()->ReportError(error, __FILE__, __LINE__, ITK_LOCATION);
}


//------------------------------------------------------------------------------
template <typename TImage>
void
OpenCLKernelToImageBridge<TImage>::SetImage(OpenCLKernel &                      kernel,
                                            const cl_uint                       argumentIndex,
                                            const typename ImageType::Pointer & image,
                                            const bool                          updateCPU)
{
  if (ImageType::ImageDimension > 3 || ImageType::ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetImage"
                             " supports only 1D/2D/3D images.");
    return;
  }

  OpenCLKernelToImageBridge<TImage>::SetImageDataManager(kernel, argumentIndex, image->GetGPUDataManager(), updateCPU);
}


//------------------------------------------------------------------------------
template <typename TImage>
void
OpenCLKernelToImageBridge<TImage>::SetImageMetaData(OpenCLKernel &                      kernel,
                                                    const cl_uint                       argumentIndex,
                                                    const typename ImageType::Pointer & image,
                                                    typename GPUDataManager::Pointer &  imageMetaDataManager)
{
  if (ImageType::ImageDimension > 3 || ImageType::ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetImageMetaData"
                             " supports only 1D/2D/3D images.");
    return;
  }

  if (imageMetaDataManager.IsNull())
  {
    itkGenericExceptionMacro(<< "The data manager is NULL."
                                "Unable to set ITK image meta data to the kernel.");
    return;
  }

  // Set image base
  imageMetaDataManager->Initialize();
  imageMetaDataManager->SetBufferFlag(CL_MEM_READ_ONLY);

  switch (static_cast<unsigned int>(ImageDimension))
  {
    case 1:
    {
      ITKOpenCLKernelToImageBridge::OCLImageBase1D imageBase1D;

      if (image.IsNotNull())
      {
        const typename ImageType::SizeType &      size = image->GetLargestPossibleRegion().GetSize();
        const typename ImageType::SpacingType &   spacing = image->GetSpacing();
        const typename ImageType::PointType &     origin = image->GetOrigin();
        const typename ImageType::DirectionType & direction = image->GetDirection();
        const typename ImageType::DirectionType & i2pp = image->GetIndexToPhysicalPoint();
        const typename ImageType::DirectionType & pp2i = image->GetPhysicalPointToIndex();

        // Set Size, Spacing, Origin
        imageBase1D.Size = size[0];
        imageBase1D.Spacing = spacing[0];
        imageBase1D.Origin = origin[0];

        // Set Directions
        imageBase1D.Direction = static_cast<float>(direction[0][0]);
        imageBase1D.IndexToPhysicalPoint = static_cast<float>(i2pp[0][0]);
        imageBase1D.PhysicalPointToIndex = static_cast<float>(pp2i[0][0]);
      }
      else
      {
        // Set Size, Spacing, Origin to zero
        imageBase1D.Size = 0;
        imageBase1D.Spacing = 0.0f;
        imageBase1D.Origin = 0.0f;

        // Set Directions to zero
        imageBase1D.Direction = 0.0f;
        imageBase1D.IndexToPhysicalPoint = 0.0f;
        imageBase1D.PhysicalPointToIndex = 0.0f;
      }

      imageMetaDataManager->SetBufferSize(sizeof(imageBase1D));
      imageMetaDataManager->Allocate();
      imageMetaDataManager->SetCPUBufferPointer(&imageBase1D);
    }
    break;
    case 2:
    {
      ITKOpenCLKernelToImageBridge::OCLImageBase2D imageBase2D;

      if (image.IsNotNull())
      {
        const typename ImageType::SizeType &      size = image->GetLargestPossibleRegion().GetSize();
        const typename ImageType::SpacingType &   spacing = image->GetSpacing();
        const typename ImageType::PointType &     origin = image->GetOrigin();
        const typename ImageType::DirectionType & direction = image->GetDirection();
        const typename ImageType::DirectionType & i2pp = image->GetIndexToPhysicalPoint();
        const typename ImageType::DirectionType & pp2i = image->GetPhysicalPointToIndex();

        // Set Size, Spacing, Origin
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          imageBase2D.Size.s[i] = size[i];
          imageBase2D.Spacing.s[i] = spacing[i];
          imageBase2D.Origin.s[i] = origin[i];
        }

        // Set Directions
        unsigned int index = 0;
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          for (unsigned int j = 0; j < ImageDimension; ++j)
          {
            imageBase2D.Direction.s[index] = static_cast<float>(direction[i][j]);
            imageBase2D.IndexToPhysicalPoint.s[index] = static_cast<float>(i2pp[i][j]);
            imageBase2D.PhysicalPointToIndex.s[index] = static_cast<float>(pp2i[i][j]);
            ++index;
          }
        }
      }
      else
      {
        // Set Size, Spacing, Origin to zero
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          imageBase2D.Size.s[i] = 0;
          imageBase2D.Spacing.s[i] = 0.0f;
          imageBase2D.Origin.s[i] = 0.0f;
        }
        // Set Directions to zero
        for (unsigned int i = 0; i < 4; ++i)
        {
          imageBase2D.Direction.s[i] = 0.0f;
          imageBase2D.IndexToPhysicalPoint.s[i] = 0.0f;
          imageBase2D.PhysicalPointToIndex.s[i] = 0.0f;
        }
      }

      imageMetaDataManager->SetBufferSize(sizeof(imageBase2D));
      imageMetaDataManager->Allocate();
      imageMetaDataManager->SetCPUBufferPointer(&imageBase2D);
    }
    break;
    case 3:
    {
      ITKOpenCLKernelToImageBridge::OCLImageBase3D imageBase3D;

      if (image.IsNotNull())
      {
        const typename ImageType::SizeType &      size = image->GetLargestPossibleRegion().GetSize();
        const typename ImageType::SpacingType &   spacing = image->GetSpacing();
        const typename ImageType::PointType &     origin = image->GetOrigin();
        const typename ImageType::DirectionType & direction = image->GetDirection();
        const typename ImageType::DirectionType & i2pp = image->GetIndexToPhysicalPoint();
        const typename ImageType::DirectionType & pp2i = image->GetPhysicalPointToIndex();

        // Set Size, Spacing, Origin
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          imageBase3D.Size.s[i] = size[i];
          imageBase3D.Spacing.s[i] = spacing[i];
          imageBase3D.Origin.s[i] = origin[i];
        }

        // Set Directions
        unsigned int index = 0;
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          for (unsigned int j = 0; j < ImageDimension; ++j)
          {
            imageBase3D.Direction.s[index] = static_cast<float>(direction[i][j]);
            imageBase3D.IndexToPhysicalPoint.s[index] = static_cast<float>(i2pp[i][j]);
            imageBase3D.PhysicalPointToIndex.s[index] = static_cast<float>(pp2i[i][j]);
            ++index;
          }
        }
        for (unsigned int i = 9; i < 16; ++i)
        {
          imageBase3D.Direction.s[i] = 0.0f;
          imageBase3D.IndexToPhysicalPoint.s[i] = 0.0f;
          imageBase3D.PhysicalPointToIndex.s[i] = 0.0f;
        }
      }
      else
      {
        // Set Size, Spacing, Origin to zero
        for (unsigned int i = 0; i < ImageDimension; ++i)
        {
          imageBase3D.Size.s[i] = 0;
          imageBase3D.Spacing.s[i] = 0.0f;
          imageBase3D.Origin.s[i] = 0.0f;
        }
        // Set Directions to zero
        for (unsigned int i = 0; i < 16; ++i)
        {
          imageBase3D.Direction.s[i] = 0.0f;
          imageBase3D.IndexToPhysicalPoint.s[i] = 0.0f;
          imageBase3D.PhysicalPointToIndex.s[i] = 0.0f;
        }
      }

      imageMetaDataManager->SetBufferSize(sizeof(imageBase3D));
      imageMetaDataManager->Allocate();
      imageMetaDataManager->SetCPUBufferPointer(&imageBase3D);
    }
    break;
    default:
      break;
  }

  imageMetaDataManager->SetGPUDirtyFlag(true);
  imageMetaDataManager->UpdateGPUBuffer();

  OpenCLKernelToImageBridge<TImage>::SetImageDataManager(kernel, argumentIndex, imageMetaDataManager, false);
}


//------------------------------------------------------------------------------
template <typename TImage>
void
OpenCLKernelToImageBridge<TImage>::SetDirection(OpenCLKernel &                            kernel,
                                                const cl_uint                             argumentIndex,
                                                const typename ImageType::DirectionType & direction)
{
  if (ImageDimension > 3 || ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetDirection"
                             " supports only 1D/2D/3D images.");
    return;
  }
  kernel.SetArg(argumentIndex, direction);
}


//------------------------------------------------------------------------------
template <typename TImage>
void
OpenCLKernelToImageBridge<TImage>::SetSize(OpenCLKernel &                       kernel,
                                           const cl_uint                        argumentIndex,
                                           const typename ImageType::SizeType & size)
{
  if (ImageDimension > 3 || ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetSize"
                             " supports only 1D/2D/3D images.");
    return;
  }
  kernel.SetArg(argumentIndex, size);
}


//------------------------------------------------------------------------------
template <typename TImage>
void
itk::OpenCLKernelToImageBridge<TImage>::SetOrigin(OpenCLKernel &                        kernel,
                                                  const cl_uint                         argumentIndex,
                                                  const typename ImageType::PointType & origin)
{
  if (ImageDimension > 3 || ImageDimension < 1)
  {
    itkGenericExceptionMacro("OpenCLKernelToImageBridge::SetOrigin"
                             " supports only 1D/2D/3D images.");
    return;
  }
  kernel.SetArg(argumentIndex, origin);
}


} // end namespace itk

#endif
