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
#ifndef itkGPUTranslationTransformBase_hxx
#define itkGPUTranslationTransformBase_hxx

#include "itkGPUTranslationTransformBase.h"
#include <iomanip>

// begin of ITKGPUTranslationTransformBase namespace
namespace ITKGPUTranslationTransformBase
{
typedef struct
{
  cl_float offset;
} GPUTranslationTransformBase1D;

typedef struct
{
  cl_float2 offset;
} GPUTranslationTransformBase2D;

typedef struct
{
  cl_float3 offset;
} GPUTranslationTransformBase3D;

//------------------------------------------------------------------------------
template <unsigned int ImageDimension>
struct SpaceDimensionToType
{};

//----------------------------------------------------------------------------
// Offset
template <typename TScalarType, unsigned int SpaceDimension>
void
SetOffset1(const itk::Vector<TScalarType, SpaceDimension> &, cl_float &, SpaceDimensionToType<SpaceDimension>)
{}

template <typename TScalarType, unsigned int SpaceDimension>
void
SetOffset2(const itk::Vector<TScalarType, SpaceDimension> &, cl_float2 &, SpaceDimensionToType<SpaceDimension>)
{}

template <typename TScalarType, unsigned int SpaceDimension>
void
SetOffset3(const itk::Vector<TScalarType, SpaceDimension> &, cl_float4 &, SpaceDimensionToType<SpaceDimension>)
{}

template <typename TScalarType>
void
SetOffset1(const itk::Vector<TScalarType, 1> & offset, cl_float & ocloffset, SpaceDimensionToType<1>)
{
  ocloffset = offset[0];
}


template <typename TScalarType>
void
SetOffset2(const itk::Vector<TScalarType, 2> & offset, cl_float2 & ocloffset, SpaceDimensionToType<2>)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 2; ++i)
  {
    ocloffset.s[id++] = offset[i];
  }
}


template <typename TScalarType>
void
SetOffset3(const itk::Vector<TScalarType, 3> & offset, cl_float4 & ocloffset, SpaceDimensionToType<3>)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    ocloffset.s[id++] = offset[i];
  }
  ocloffset.s[3] = 0.0;
}


} // namespace ITKGPUTranslationTransformBase

//------------------------------------------------------------------------------
namespace itk
{
template <typename TScalarType, unsigned int NDimensions>
GPUTranslationTransformBase<TScalarType, NDimensions>::GPUTranslationTransformBase()
{
  // Add GPUTranslationTransformBase source
  const std::string sourcePath(GPUTranslationTransformBaseKernel::GetOpenCLSource());
  m_Sources.push_back(sourcePath);

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag(CL_MEM_READ_ONLY);

  using namespace ITKGPUTranslationTransformBase;
  const unsigned int Dimension = SpaceDimension;

  switch (Dimension)
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUTranslationTransformBase1D));
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUTranslationTransformBase2D));
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUTranslationTransformBase3D));
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
} // end Constructor


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
GPUDataManager::Pointer
GPUTranslationTransformBase<TScalarType, NDimensions>::GetParametersDataManager() const
{
  using namespace ITKGPUTranslationTransformBase;
  const SpaceDimensionToType<SpaceDimension> dim = {};
  const unsigned int                         Dimension = SpaceDimension;

  switch (Dimension)
  {
    case 1:
    {
      GPUTranslationTransformBase1D translationBase;
      SetOffset1<ScalarType>(GetCPUOffset(), translationBase.offset, dim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&translationBase);
    }
    break;
    case 2:
    {
      GPUTranslationTransformBase2D translationBase;
      SetOffset2<ScalarType>(GetCPUOffset(), translationBase.offset, dim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&translationBase);
    }
    break;
    case 3:
    {
      GPUTranslationTransformBase3D translationBase;
      SetOffset3<ScalarType>(GetCPUOffset(), translationBase.offset, dim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&translationBase);
    }
    break;
    default:
      break;
  }

  this->m_ParametersDataManager->SetGPUDirtyFlag(true);
  this->m_ParametersDataManager->UpdateGPUBuffer();

  return this->m_ParametersDataManager;
} // end GetParametersDataManager()


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NDimensions>
bool
GPUTranslationTransformBase<TScalarType, NDimensions>::GetSourceCode(std::string & source) const
{
  if (this->m_Sources.empty())
  {
    return false;
  }

  // Create the final source code
  std::ostringstream sources;
  // Add other sources
  for (std::size_t i = 0; i < this->m_Sources.size(); ++i)
  {
    sources << this->m_Sources[i] << std::endl;
  }
  source = sources.str();
  return true;
} // end GetSourceCode()


} // end namespace itk

#endif /* itkGPUTranslationTransformBase_hxx */
