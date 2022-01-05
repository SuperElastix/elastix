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
#ifndef itkGPUEuler2DTransformFactory_hxx
#define itkGPUEuler2DTransformFactory_hxx

#include "itkGPUEuler2DTransformFactory.h"

namespace itk
{
template <typename NDimensions>
void
GPUEuler2DTransformFactory2<NDimensions>::RegisterOneFactory()
{
  using GPUTransformFactoryType = GPUEuler2DTransformFactory2<NDimensions>;
  auto factory = GPUTransformFactoryType::New();
  ObjectFactoryBase::RegisterFactory(factory);
}


//------------------------------------------------------------------------------
template <typename NDimensions>
GPUEuler2DTransformFactory2<NDimensions>::GPUEuler2DTransformFactory2()
{
  this->RegisterAll();
}


//------------------------------------------------------------------------------
template <typename NDimensions>
void
GPUEuler2DTransformFactory2<NDimensions>::Register2D()
{
  // Define visitor and perform factory registration
  typelist::Visit<RealTypeList> visitor;
  visitor(*this);
}


} // namespace itk

#endif // end #ifndef itkGPUEuler2DTransformFactory_hxx
