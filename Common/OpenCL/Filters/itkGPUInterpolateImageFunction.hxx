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
#ifndef itkGPUInterpolateImageFunction_hxx
#define itkGPUInterpolateImageFunction_hxx

#include "itkGPUInterpolateImageFunction.h"
#include "itkOpenCLOstreamSupport.h"

// begin of unnamed namespace
namespace
{
typedef struct
{
  cl_uint  start_index;
  cl_uint  end_index;
  cl_float start_continuous_index;
  cl_float end_continuous_index;
} GPUImageFunction1D;

typedef struct
{
  cl_uint2  start_index;
  cl_uint2  end_index;
  cl_float2 start_continuous_index;
  cl_float2 end_continuous_index;
} GPUImageFunction2D;

typedef struct
{
  cl_uint3  start_index;
  cl_uint3  end_index;
  cl_float3 start_continuous_index;
  cl_float3 end_continuous_index;
} GPUImageFunction3D;

//------------------------------------------------------------------------------

// IndexType
template <typename ImageType>
void
SetIndex(const typename ImageType::IndexType index, cl_uint & oclindex)
{
  oclindex = index[0];
}


template <typename ImageType>
void
SetIndex(const typename ImageType::IndexType index, cl_uint2 & oclindex)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 2; ++i)
  {
    oclindex.s[id++] = index[i];
  }
}


template <typename ImageType>
void
SetIndex(const typename ImageType::IndexType index, cl_uint4 & oclindex)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    oclindex.s[id++] = index[i];
  }
  oclindex.s[3] = 0;
}


// ContinuousIndexType
template <typename TContinuousIndex>
void
SetContinuousIndex(const TContinuousIndex & cindex, cl_float & oclindex)
{
  oclindex = cindex[0];
}


template <typename TContinuousIndex>
void
SetContinuousIndex(const TContinuousIndex & cindex, cl_float2 & oclindex)
{
  oclindex.s[0] = cindex[0];
  oclindex.s[1] = cindex[1];
}


template <typename TContinuousIndex>
void
SetContinuousIndex(const TContinuousIndex & cindex, cl_float4 & oclindex)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    oclindex.s[id++] = cindex[i];
  }
  oclindex.s[3] = 0.0;
}


} // end of unnamed namespace

//------------------------------------------------------------------------------
namespace itk
{
template <typename TInputImage, typename TCoordRep, typename TParentInterpolateImageFunction>
GPUInterpolateImageFunction<TInputImage, TCoordRep, TParentInterpolateImageFunction>::GPUInterpolateImageFunction()
{
  const unsigned int ImageDim = InputImageType::ImageDimension;

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag(CL_MEM_READ_ONLY);

  switch (ImageDim)
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUImageFunction1D));
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUImageFunction2D));
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUImageFunction3D));
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TParentInterpolateImageFunction>
GPUDataManager::Pointer
GPUInterpolateImageFunction<TInputImage, TCoordRep, TParentInterpolateImageFunction>::GetParametersDataManager() const
{
  const unsigned int ImageDim = InputImageType::ImageDimension;

  switch (ImageDim)
  {
    case 1:
    {
      GPUImageFunction1D imageFunction;

      SetIndex<InputImageType>(this->m_StartIndex, imageFunction.start_index);
      SetIndex<InputImageType>(this->m_EndIndex, imageFunction.end_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_StartContinuousIndex, imageFunction.start_continuous_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_EndContinuousIndex, imageFunction.end_continuous_index);
      this->m_ParametersDataManager->SetCPUBufferPointer(&imageFunction);
    }
    break;
    case 2:
    {
      GPUImageFunction2D imageFunction;

      SetIndex<InputImageType>(this->m_StartIndex, imageFunction.start_index);
      SetIndex<InputImageType>(this->m_EndIndex, imageFunction.end_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_StartContinuousIndex, imageFunction.start_continuous_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_EndContinuousIndex, imageFunction.end_continuous_index);
      this->m_ParametersDataManager->SetCPUBufferPointer(&imageFunction);
    }
    break;
    case 3:
    {
      GPUImageFunction3D imageFunction;

      SetIndex<InputImageType>(this->m_StartIndex, imageFunction.start_index);
      SetIndex<InputImageType>(this->m_EndIndex, imageFunction.end_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_StartContinuousIndex, imageFunction.start_continuous_index);
      SetContinuousIndex<ContinuousIndexType>(this->m_EndContinuousIndex, imageFunction.end_continuous_index);
      this->m_ParametersDataManager->SetCPUBufferPointer(&imageFunction);
    }
    break;
    default:
      break;
  }

  this->m_ParametersDataManager->SetGPUDirtyFlag(true);
  this->m_ParametersDataManager->UpdateGPUBuffer();

  return this->m_ParametersDataManager;
}


//------------------------------------------------------------------------------
template <typename TInputImage, typename TCoordRep, typename TParentInterpolateImageFunction>
void
GPUInterpolateImageFunction<TInputImage, TCoordRep, TParentInterpolateImageFunction>::PrintSelf(std::ostream & os,
                                                                                                Indent indent) const
{
  CPUSuperclass::PrintSelf(os, indent);
}


} // end namespace itk

#endif /* itkGPUInterpolateImageFunction_hxx */
