/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/
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
  itkTypeMacro(GPUTransformBase, Object);

  /** */
  virtual bool GetSourceCode(std::string &_source) const;

  /** */
  virtual GPUDataManager::Pointer GetParametersDataManager() const;

protected:
  GPUTransformBase();
  virtual ~GPUTransformBase() {};

  GPUDataManager::Pointer m_ParametersDataManager;
};

} // end namespace itk

#endif /* __itkGPUTransformBase_h */
