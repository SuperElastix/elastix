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
#ifndef itkGPUMatrixOffsetTransformBase_hxx
#define itkGPUMatrixOffsetTransformBase_hxx

#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

// begin of ITKGPUMatrixOffsetTransformBase namespace
namespace ITKGPUMatrixOffsetTransformBase
{
typedef struct
{
  cl_float matrix;
  cl_float offset;
} GPUMatrixOffsetTransformBase1D;

typedef struct
{
  cl_float4 matrix;
  cl_float2 offset;
} GPUMatrixOffsetTransformBase2D;

typedef struct
{
  cl_float16 matrix; // OpenCL does not have float9
  cl_float3  offset;
} GPUMatrixOffsetTransformBase3D;

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
SetOffset3(const itk::Vector<TScalarType, SpaceDimension> &, cl_float3 &, SpaceDimensionToType<SpaceDimension>)
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
SetOffset3(const itk::Vector<TScalarType, 3> & offset, cl_float3 & ocloffset, SpaceDimensionToType<3>)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    ocloffset.s[id++] = offset[i];
  }
  ocloffset.s[3] = 0.0;
}


//----------------------------------------------------------------------------
// Matrix
template <typename TScalarType, unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension>
void
SetMatrix1(const itk::Matrix<TScalarType, OutputSpaceDimension, InputSpaceDimension> &,
           cl_float &,
           SpaceDimensionToType<OutputSpaceDimension>,
           SpaceDimensionToType<InputSpaceDimension>)
{}

template <typename TScalarType, unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension>
void
SetMatrix2(const itk::Matrix<TScalarType, OutputSpaceDimension, InputSpaceDimension> &,
           cl_float4 &,
           SpaceDimensionToType<OutputSpaceDimension>,
           SpaceDimensionToType<InputSpaceDimension>)
{}

template <typename TScalarType, unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension>
void
SetMatrix3(const itk::Matrix<TScalarType, OutputSpaceDimension, InputSpaceDimension> &,
           cl_float16 &,
           SpaceDimensionToType<OutputSpaceDimension>,
           SpaceDimensionToType<InputSpaceDimension>)
{}

template <typename TScalarType>
void
SetMatrix1(const itk::Matrix<TScalarType, 1, 1> & matrix,
           cl_float &                             oclmatrix,
           SpaceDimensionToType<1>,
           SpaceDimensionToType<1>)
{
  oclmatrix = matrix[0][0];
}


template <typename TScalarType>
void
SetMatrix2(const itk::Matrix<TScalarType, 2, 2> & matrix,
           cl_float4 &                            oclmatrix,
           SpaceDimensionToType<2>,
           SpaceDimensionToType<2>)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 2; ++i)
  {
    for (unsigned int j = 0; j < 2; ++j)
    {
      oclmatrix.s[id++] = matrix[i][j];
    }
  }
}


template <typename TScalarType>
void
SetMatrix3(const itk::Matrix<TScalarType, 3, 3> & matrix,
           cl_float16 &                           oclmatrix,
           SpaceDimensionToType<3>,
           SpaceDimensionToType<3>)
{
  unsigned int id = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < 3; ++j)
    {
      oclmatrix.s[id++] = matrix[i][j];
    }
  }
  for (unsigned int i = 9; i < 16; ++i)
  {
    oclmatrix.s[i] = 0.0;
  }
}


} // namespace ITKGPUMatrixOffsetTransformBase

//------------------------------------------------------------------------------
namespace itk
{
template <typename TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
GPUMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GPUMatrixOffsetTransformBase()
{
  // Add GPUMatrixOffsetTransformBase source
  const std::string sourcePath(GPUMatrixOffsetTransformBaseKernel::GetOpenCLSource());
  this->m_Sources.push_back(sourcePath);

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag(CL_MEM_READ_ONLY);

  using namespace ITKGPUMatrixOffsetTransformBase;
  const unsigned int OutputDimension = OutputSpaceDimension;

  switch (OutputDimension)
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase1D));
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase2D));
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize(sizeof(GPUMatrixOffsetTransformBase3D));
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
}


//------------------------------------------------------------------------------
template <typename TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
GPUDataManager::Pointer
GPUMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetParametersDataManager() const
{
  using namespace ITKGPUMatrixOffsetTransformBase;
  const unsigned int                               OutputDimension = OutputSpaceDimension;
  const SpaceDimensionToType<InputSpaceDimension>  idim = {};
  const SpaceDimensionToType<OutputSpaceDimension> odim = {};

  switch (OutputDimension)
  {
    case 1:
    {
      GPUMatrixOffsetTransformBase1D transformBase;
      SetMatrix1<ScalarType>(GetCPUMatrix(), transformBase.matrix, odim, idim);
      SetOffset1<ScalarType>(GetCPUOffset(), transformBase.offset, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
    }
    break;
    case 2:
    {
      GPUMatrixOffsetTransformBase2D transformBase;
      SetMatrix2<ScalarType>(GetCPUMatrix(), transformBase.matrix, odim, idim);
      SetOffset2<ScalarType>(GetCPUOffset(), transformBase.offset, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
    }
    break;
    case 3:
    {
      GPUMatrixOffsetTransformBase3D transformBase;
      SetMatrix3<ScalarType>(GetCPUMatrix(), transformBase.matrix, odim, idim);
      SetOffset3<ScalarType>(GetCPUOffset(), transformBase.offset, odim);
      this->m_ParametersDataManager->SetCPUBufferPointer(&transformBase);
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
template <typename TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
bool
GPUMatrixOffsetTransformBase<TScalarType, NInputDimensions, NOutputDimensions>::GetSourceCode(
  std::string & source) const
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
}


} // end namespace itk

#endif /* itkGPUMatrixOffsetTransformBase_hxx */
