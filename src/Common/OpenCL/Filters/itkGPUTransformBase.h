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

#include "itkGPUDataManager.h"

namespace itk
{
/** \class GPUTransformBase
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
class ITK_EXPORT GPUTransformBase
{
public:
  /** Run-time type information (and related methods). */
  virtual const char * GetNameOfClass() const { return "GPUTransformBase"; }

  /** */
  virtual bool GetSourceCode( std::string & _source ) const;

  /** */
  virtual bool IsIdentityTransform() const { return false; }
  virtual bool IsMatrixOffsetTransform() const { return false; }
  virtual bool IsTranslationTransform() const { return false; }
  virtual bool IsBSplineTransform() const { return false; }

  /** */
  virtual GPUDataManager::Pointer GetParametersDataManager() const;

  /** */
  virtual GPUDataManager::Pointer GetParametersDataManager( const std::size_t index ) const;

protected:
  GPUTransformBase();
  virtual ~GPUTransformBase() {}

  GPUDataManager::Pointer m_ParametersDataManager;
};
} // end namespace itk

#endif /* __itkGPUTransformBase_h */
