/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUInterpolatorBase_h
#define __itkGPUInterpolatorBase_h

#include "itkMacro.h"
#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUInterpolatorBase
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
class ITK_EXPORT GPUInterpolatorBase
{
public:
  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUInterpolatorBase, Object );

  /** */
  virtual bool GetSourceCode( std::string & _source ) const;

  /** */
  virtual GPUDataManager::Pointer GetParametersDataManager() const;

protected:
  GPUInterpolatorBase();
  virtual ~GPUInterpolatorBase() {}

  GPUDataManager::Pointer m_ParametersDataManager;
};
} // end namespace itk

#endif /* __itkGPUInterpolatorBase_h */
