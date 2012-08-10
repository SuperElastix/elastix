/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUMatrixOffsetTransformBase_h
#define __itkGPUMatrixOffsetTransformBase_h

#include "itkMacro.h"

namespace itk
{
/** \class GPUMatrixOffsetTransformBase
 */
/** Create a helper GPU Kernel class for itkGPUMatrixOffsetTransformBase */
itkGPUKernelClassMacro(GPUMatrixOffsetTransformBaseHeaderKernel);
itkGPUKernelClassMacro(GPUMatrixOffsetTransformBaseKernel);

} // end namespace itk

#endif /* itkGPUMatrixOffsetTransformBase_h */
