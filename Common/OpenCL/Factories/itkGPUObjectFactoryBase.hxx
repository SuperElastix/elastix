/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkGPUObjectFactoryBase_hxx
#define itkGPUObjectFactoryBase_hxx

#include "itkGPUObjectFactoryBase.h"
#include "itkOpenCLContext.h"

namespace itk
{
template <typename NDimensions>
void
GPUObjectFactoryBase<NDimensions>::RegisterAll()
{
  OpenCLContext::Pointer context = OpenCLContext::GetInstance();
  if (context->IsCreated())
  {
    if (Support1D)
    {
      this->Register1D();
    }
    if (Support2D)
    {
      this->Register2D();
    }
    if (Support3D)
    {
      this->Register3D();
    }
  }
}


} // namespace itk

#endif // end #ifndef itkGPUObjectFactoryBase_hxx
