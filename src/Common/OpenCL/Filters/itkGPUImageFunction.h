/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUImageFunction_h
#define __itkGPUImageFunction_h

#include "itkMacro.h"

namespace itk
{
/** \class GPUImageFunction
 */
/** Create a helper GPU Kernel class for itkGPUImageFunction */
itkGPUKernelClassMacro(GPUImageFunctionKernel);

} // end namespace itk

#endif /* itkGPUImageFunction_h */
