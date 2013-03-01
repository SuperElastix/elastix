/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// This implementation was taken from elastix (http://elastix.isi.uu.nl/).
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//

#ifndef __itkGPUExplicitSynchronization_h
#define __itkGPUExplicitSynchronization_h

#include "itkGPUImage.h"

namespace itk
{
//------------------------------------------------------------------------------
// GPU explicit synchronization helper function
template< class ImageToImageFilterType, class OutputImageType >
void GPUExplicitSync( typename ImageToImageFilterType::Pointer & filter,
                      const bool filterUpdate = true,
                      const bool releaseGPUMemory = false )
{
  if( filter.IsNotNull() )
  {
    if ( filterUpdate )
    {
      filter->Update();
    }

    typedef typename OutputImageType::PixelType                               OutputImagePixelType;
    typedef GPUImage< OutputImagePixelType, OutputImageType::ImageDimension > GPUOutputImageType;
    GPUOutputImageType *GPUOutput = dynamic_cast< GPUOutputImageType * >( filter->GetOutput() );
    if ( GPUOutput )
    {
      GPUOutput->UpdateBuffers();
    }

    if ( releaseGPUMemory )
    {
      GPUOutput->GetGPUDataManager()->Initialize();
    }
  }
  else
  {
    itkGenericExceptionMacro( << "The filter pointer is null." );
  }
}

//------------------------------------------------------------------------------
// GPUImage explicit synchronization helper function
template< class ImageType >
void GPUImageSyncPixelContainer( typename ImageType::Pointer & image )
{
  if( image.IsNotNull() )
  {
    typedef typename ImageType::PixelType ImagePixelType;
    typedef GPUImage< ImagePixelType, ImageType::ImageDimension > GPUImageType;
    GPUImageType *gpuImage = dynamic_cast< GPUImageType * >( image.GetPointer() );
    if ( gpuImage )
    {
      gpuImage->AllocateGPU();
    }
  }
  else
  {
    itkGenericExceptionMacro( << "The image pointer is null." );
  }
}

}

#endif /* __itkGPUExplicitSynchronization_h */
