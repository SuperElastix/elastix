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
#ifndef itkGPUCastImageFilterFactory_hxx
#define itkGPUCastImageFilterFactory_hxx

#include "itkGPUCastImageFilterFactory.h"

namespace itk
{
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::RegisterOneFactory()
{
  using GPUFilterFactoryType = GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>;
  auto factory = GPUFilterFactoryType::New();
  ObjectFactoryBase::RegisterFactory(factory);
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::GPUCastImageFilterFactory2()
{
  this->RegisterAll();
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register1D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 1> visitor0;
  visitor0(*this);

  // Perform extra factory registration with float and double types
  const bool inputHasFloat = typelist::HasType<TTypeListIn, float>::Type;
  const bool inputHasDouble = typelist::HasType<TTypeListIn, double>::Type;
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  const bool outputHasDouble = typelist::HasType<TTypeListOut, double>::Type;

  if (!inputHasFloat || !outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<FloatTypeList, FloatTypeList, 1> visitor1;
    visitor1(*this);
  }

  if (!inputHasFloat || !outputHasDouble)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    typelist::DualVisitDimension<FloatTypeList, DoubleTypeList, 1> visitor2;
    visitor2(*this);
  }

  if (!inputHasDouble || !outputHasFloat)
  {
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<DoubleTypeList, FloatTypeList, 1> visitor3;
    visitor3(*this);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register2D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 2> visitor0;
  visitor0(*this);

  // Perform extra factory registration with float and double types
  const bool inputHasFloat = typelist::HasType<TTypeListIn, float>::Type;
  const bool inputHasDouble = typelist::HasType<TTypeListIn, double>::Type;
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  const bool outputHasDouble = typelist::HasType<TTypeListOut, double>::Type;

  if (!inputHasFloat || !outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<FloatTypeList, FloatTypeList, 2> visitor1;
    visitor1(*this);
  }

  if (!inputHasFloat || !outputHasDouble)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    typelist::DualVisitDimension<FloatTypeList, DoubleTypeList, 2> visitor2;
    visitor2(*this);
  }

  if (!inputHasDouble || !outputHasFloat)
  {
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<DoubleTypeList, FloatTypeList, 2> visitor3;
    visitor3(*this);
  }
}


//------------------------------------------------------------------------------
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
void
GPUCastImageFilterFactory2<TTypeListIn, TTypeListOut, NDimensions>::Register3D()
{
  // Define visitor and perform factory registration
  typelist::DualVisitDimension<TTypeListIn, TTypeListOut, 3> visitor0;
  visitor0(*this);

  // Perform extra factory registration with float and double types
  const bool inputHasFloat = typelist::HasType<TTypeListIn, float>::Type;
  const bool inputHasDouble = typelist::HasType<TTypeListIn, double>::Type;
  const bool outputHasFloat = typelist::HasType<TTypeListOut, float>::Type;
  const bool outputHasDouble = typelist::HasType<TTypeListOut, double>::Type;

  if (!inputHasFloat || !outputHasFloat)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<FloatTypeList, FloatTypeList, 3> visitor1;
    visitor1(*this);
  }

  if (!inputHasFloat || !outputHasDouble)
  {
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    typelist::DualVisitDimension<FloatTypeList, DoubleTypeList, 3> visitor2;
    visitor2(*this);
  }

  if (!inputHasDouble || !outputHasFloat)
  {
    using DoubleTypeList = typelist::MakeTypeList<double>::Type;
    using FloatTypeList = typelist::MakeTypeList<float>::Type;
    typelist::DualVisitDimension<DoubleTypeList, FloatTypeList, 3> visitor3;
    visitor3(*this);
  }
}


} // namespace itk

#endif // end #ifndef itkGPUCastImageFilterFactory_hxx
