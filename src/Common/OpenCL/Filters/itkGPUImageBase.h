/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUImageBase_h
#define __itkGPUImageBase_h

#include "itkMacro.h"

namespace itk
{
/** \class GPUImageBase
 */
/** Create a helper GPU Kernel class for itkGPUImageBase */
itkGPUKernelClassMacro(GPUImageBaseKernel);

} // end namespace itk

#endif /* itkGPUImageBase_h */
