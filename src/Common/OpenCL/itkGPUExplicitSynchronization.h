/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUExplicitSynchronization_h
#define __itkGPUExplicitSynchronization_h

#include "itkGPUImage.h"

namespace itk
{
  //------------------------------------------------------------------------------
  // GPU explicit synchronization helper function
  template<class ImageToImageFilterType, class OutputImageType>
  void GPUExplicitSync(typename ImageToImageFilterType::Pointer &filter,
    const bool filterUpdate = true,
    const bool releaseGPUMemory = false)
  {
    if(filterUpdate)
    {
      filter->Update();
    }

    typedef typename OutputImageType::PixelType OutputImagePixelType;
    typedef itk::GPUImage<OutputImagePixelType, OutputImageType::ImageDimension> GPUOutputImageType;
    GPUOutputImageType *GPUOutput = dynamic_cast<GPUOutputImageType *>(filter->GetOutput());
    if(GPUOutput)
    {
      GPUOutput->UpdateBuffers();
    }

    if(releaseGPUMemory)
    {
      GPUOutput->GetGPUDataManager()->Initialize();
    }
  }
}

#endif // end #ifndef __LoggerHelper_h
