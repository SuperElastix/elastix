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
#ifndef __itkGPUInterpolatorCopier_hxx
#define __itkGPUInterpolatorCopier_hxx

#include "itkGPUInterpolatorCopier.h"

// ITK CPU interpolators
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkBSplineInterpolateImageFunction.h"

// ITK GPU interpolators
#include "itkGPUNearestNeighborInterpolateImageFunction.h"
#include "itkGPULinearInterpolateImageFunction.h"
#include "itkGPUBSplineInterpolateImageFunction.h"

// GPU factory include
#include "itkGPUImageFactory.h"

namespace itk
{
//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TInterpolator, typename TOutputCoordRep >
GPUInterpolatorCopier< TTypeList, NDimensions, TInterpolator, TOutputCoordRep >
::GPUInterpolatorCopier()
{
  this->m_InputInterpolator     = NULL;
  this->m_Output                = NULL;
  this->m_ExplicitOutput        = NULL;
  this->m_InternalTransformTime = 0;
  this->m_ExplicitMode          = true;
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TInterpolator, typename TOutputCoordRep >
void
GPUInterpolatorCopier< TTypeList, NDimensions, TInterpolator, TOutputCoordRep >
::Update( void )
{
  if( !this->m_InputInterpolator )
  {
    itkExceptionMacro( << "Input Interpolator has not been connected" );
    return;
  }

  // Update only if the input AdvancedCombinationTransform has been modified
  const ModifiedTimeType t = this->m_InputInterpolator->GetMTime();

  if( t == this->m_InternalTransformTime )
  {
    return; // No need to update
  }
  else if( t > this->m_InternalTransformTime )
  {
    // Cache the timestamp
    this->m_InternalTransformTime = t;

    // Try Nearest
    typedef NearestNeighborInterpolateImageFunction<
      CPUInputImageType, CPUCoordRepType > NearestNeighborInterpolatorType;
    const typename NearestNeighborInterpolatorType::ConstPointer nearest
      = dynamic_cast< const NearestNeighborInterpolatorType * >( m_InputInterpolator.GetPointer() );

    if( nearest )
    {
      if( this->m_ExplicitMode )
      {
        // Create GPU NearestNeighbor interpolator in explicit mode
        typedef GPUNearestNeighborInterpolateImageFunction<
          GPUInputImageType, GPUCoordRepType > GPUNearestNeighborInterpolatorType;
        this->m_ExplicitOutput = GPUNearestNeighborInterpolatorType::New();
      }
      else
      {
        // Create GPU NearestNeighbor interpolator in implicit mode
        typedef NearestNeighborInterpolateImageFunction<
          CPUInputImageType, GPUCoordRepType > GPUNearestNeighborInterpolatorType;
        this->m_Output = GPUNearestNeighborInterpolatorType::New();
      }
      return;
    }

    // Try Linear
    typedef LinearInterpolateImageFunction<
      CPUInputImageType, CPUCoordRepType > LinearInterpolatorType;
    const typename LinearInterpolatorType::ConstPointer linear
      = dynamic_cast< const LinearInterpolatorType * >( m_InputInterpolator.GetPointer() );

    if( linear )
    {
      if( this->m_ExplicitMode )
      {
        // Create GPU Linear interpolator in explicit mode
        typedef GPULinearInterpolateImageFunction<
          GPUInputImageType, GPUCoordRepType > GPULinearInterpolatorType;
        this->m_ExplicitOutput = GPULinearInterpolatorType::New();
      }
      else
      {
        // Create GPU Linear interpolator in implicit mode
        typedef LinearInterpolateImageFunction<
          CPUInputImageType, GPUCoordRepType > GPULinearInterpolatorType;
        this->m_Output = GPULinearInterpolatorType::New();
      }
      return;
    }

    // Try BSpline
    typedef BSplineInterpolateImageFunction<
      CPUInputImageType, CPUCoordRepType, CPUCoordRepType > BSplineInterpolatorType;
    const typename BSplineInterpolatorType::ConstPointer bspline
      = dynamic_cast< const BSplineInterpolatorType * >( m_InputInterpolator.GetPointer() );

    if( bspline )
    {
      if( this->m_ExplicitMode )
      {
        // Register image factory because BSplineInterpolateImageFunction
        // using m_Coefficients as ITK images inside the implementation
        typedef itk::GPUImageFactory2< TTypeList, NDimensions > GPUImageFactoryType;
        typedef typename GPUImageFactoryType::Pointer           GPUImageFactoryPointer;
        GPUImageFactoryPointer imageFactory = GPUImageFactoryType::New();
        itk::ObjectFactoryBase::RegisterFactory( imageFactory );

        // Create GPU BSpline interpolator in explicit mode
        typedef GPUBSplineInterpolateImageFunction<
          GPUInputImageType, GPUCoordRepType, GPUCoordRepType > GPUBSplineInterpolatorType;
        typename GPUBSplineInterpolatorType::Pointer bsplineInterpolator
          = GPUBSplineInterpolatorType::New();
        bsplineInterpolator->SetSplineOrder( bspline->GetSplineOrder() );

        // UnRegister image factory
        itk::ObjectFactoryBase::UnRegisterFactory( imageFactory );

        this->m_ExplicitOutput = bsplineInterpolator;
      }
      else
      {
        // Create GPU BSpline interpolator in implicit mode
        typedef BSplineInterpolateImageFunction<
          CPUInputImageType, GPUCoordRepType, GPUCoordRepType > GPUBSplineInterpolatorType;
        typename GPUBSplineInterpolatorType::Pointer bsplineInterpolator
          = GPUBSplineInterpolatorType::New();
        bsplineInterpolator->SetSplineOrder( bspline->GetSplineOrder() );
        this->m_Output = bsplineInterpolator;
      }
      return;
    }

    if( this->m_Output.IsNull() )
    {
      itkExceptionMacro( << "GPUInterpolatorCopier was unable to copy interpolator from: " << this->m_InputInterpolator );
    }
  }
}


//------------------------------------------------------------------------------
template< typename TTypeList, typename NDimensions, typename TInterpolator, typename TOutputCoordRep >
void
GPUInterpolatorCopier< TTypeList, NDimensions, TInterpolator, TOutputCoordRep >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os << indent << "Input Interpolator: " << this->m_InputInterpolator << std::endl;
  os << indent << "Output Non Explicit Interpolator: " << this->m_Output << std::endl;
  os << indent << "Output Explicit Interpolator: " << this->m_ExplicitOutput << std::endl;
  os << indent << "Internal Transform Time: " << this->m_InternalTransformTime << std::endl;
  os << indent << "Explicit Mode: " << this->m_ExplicitMode << std::endl;
}


} // end namespace itk

#endif
