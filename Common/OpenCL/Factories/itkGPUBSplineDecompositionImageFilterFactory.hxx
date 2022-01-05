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
#ifndef itkGPUBSplineDecompositionImageFilterFactory_hxx
#define itkGPUBSplineDecompositionImageFilterFactory_hxx

#include "itkGPUBSplineDecompositionImageFilterFactory.h"

namespace itk
{
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::RegisterOneFactory()
{
  using GPUFilterFactoryType = GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>;
  auto factory = GPUFilterFactoryType::New();
  ObjectFactoryBase::RegisterFactory(factory);
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::
  GPUBSplineDecompositionImageFilterFactory2()
{
  this->RegisterAll();
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register1D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 1> visitor;
  visitor(*this);

  // Perform extra factory registration with float for the output type
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  if (!outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<TTypeListIn, FloatTypeList, 1> visitor1;
    visitor1(*this);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register2D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 2> visitor;
  visitor(*this);

  // Perform extra factory registration with float for the output type
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  if (!outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<TTypeListIn, FloatTypeList, 2> visitor1;
    visitor1(*this);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUBSplineDecompositionImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register3D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 3> visitor;
  visitor(*this);

  // Perform extra factory registration with float for the output type
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  if (!outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<TTypeListIn, FloatTypeList, 3> visitor1;
    visitor1(*this);
  }
}


} // namespace itk

#endif // end #ifndef itkGPUBSplineDecompositionImageFilterFactory_hxx
