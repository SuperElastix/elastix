/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#include "itkGPUTransformBase.h"

namespace itk
{
GPUTransformBase::GPUTransformBase()
{
  this->m_ParametersDataManager = GPUDataManager::New();
}

//------------------------------------------------------------------------------
bool GPUTransformBase::GetSourceCode(std::string &_source) const
{
  // do nothing here
  return true;
}

//------------------------------------------------------------------------------
GPUDataManager::Pointer GPUTransformBase::GetParametersDataManager() const
{
  return this->m_ParametersDataManager;
}

} // end namespace itk
