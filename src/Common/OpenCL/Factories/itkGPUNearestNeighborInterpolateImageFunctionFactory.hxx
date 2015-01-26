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
#ifndef __itkGPUNearestNeighborInterpolateImageFunctionFactory_hxx
#define __itkGPUNearestNeighborInterpolateImageFunctionFactory_hxx

#include "itkGPUNearestNeighborInterpolateImageFunctionFactory.h"

namespace itk
{
template< typename TTypeList, typename NDimentions >
void
GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions >
::RegisterOneFactory()
{
  typedef GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions > GPUInterpolateFactoryType;
  typename GPUInterpolateFactoryType::Pointer factory = GPUInterpolateFactoryType::New();
  ObjectFactoryBase::RegisterFactory( factory );
}


//------------------------------------------------------------------------------

template< typename TTypeList, typename NDimentions >
GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions >
::GPUNearestNeighborInterpolateImageFunctionFactory2()
{
  this->RegisterAll();
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimentions >
void
GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions >
::Register1D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension< TTypeList, 1 > visitor0;
  visitor0( *this );

  // Perform extra factory registration with float type
  const bool inputHasFloat = typelist::HasType< TTypeList, float >::Type;
  if( !inputHasFloat )
  {
    typedef typelist::MakeTypeList< float >::Type FloatTypeList;
    typelist::VisitDimension< FloatTypeList, 1 > visitor1;
    visitor1( *this );
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimentions >
void
GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions >
::Register2D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension< TTypeList, 2 > visitor0;
  visitor0( *this );

  // Perform extra factory registration with float type
  const bool inputHasFloat = typelist::HasType< TTypeList, float >::Type;
  if( !inputHasFloat )
  {
    typedef typelist::MakeTypeList< float >::Type FloatTypeList;
    typelist::VisitDimension< FloatTypeList, 2 > visitor1;
    visitor1( *this );
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimentions >
void
GPUNearestNeighborInterpolateImageFunctionFactory2< TTypeList, NDimentions >
::Register3D()
{
  // Define visitor and perform factory registration
  typelist::VisitDimension< TTypeList, 3 > visitor0;
  visitor0( *this );

  // Perform extra factory registration with float type
  const bool inputHasFloat = typelist::HasType< TTypeList, float >::Type;
  if( !inputHasFloat )
  {
    typedef typelist::MakeTypeList< float >::Type FloatTypeList;
    typelist::VisitDimension< FloatTypeList, 3 > visitor1;
    visitor1( *this );
  }
}


} // namespace itk

#endif // end #ifndef __itkGPUNearestNeighborInterpolateImageFunctionFactory_hxx
