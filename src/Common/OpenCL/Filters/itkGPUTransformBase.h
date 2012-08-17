/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUTransformBase_h
#define __itkGPUTransformBase_h

#include "itkMacro.h"
#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUTransformBase
*/
class ITK_EXPORT GPUTransformBase
{
public:
  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUTransformBase, Object );

  /** */
  virtual bool GetSourceCode( std::string & _source ) const;

  /** */
  virtual GPUDataManager::Pointer GetParametersDataManager() const;

protected:
  GPUTransformBase();
  virtual ~GPUTransformBase() {}

  GPUDataManager::Pointer m_ParametersDataManager;
};

} // end namespace itk

#endif /* __itkGPUTransformBase_h */
