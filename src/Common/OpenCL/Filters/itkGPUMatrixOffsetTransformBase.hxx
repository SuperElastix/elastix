/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUMatrixOffsetTransformBase_hxx
#define __itkGPUMatrixOffsetTransformBase_hxx

#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

// begin of ITKGPUMatrixOffsetTransformBase namespace
namespace ITKGPUMatrixOffsetTransformBase
{
typedef struct {
  cl_float matrix;
  cl_float offset;
  cl_float inverse_matrix;
} GPUMatrixOffsetTransformBase1D;

typedef struct {
  cl_float4 matrix;
  cl_float2 offset;
  cl_float4 inverse_matrix;
} GPUMatrixOffsetTransformBase2D;

typedef struct {
  cl_float16 matrix;          // OpenCL does not have float9
  cl_float3  offset;
  cl_float16 inverse_matrix;  // OpenCL does not have float9
} GPUMatrixOffsetTransformBase3D;

//------------------------------------------------------------------------------
template< unsigned int ImageDimension >
struct SpaceDimensionToType {};

//----------------------------------------------------------------------------
// Offset
template< class TScalarType, unsigned int SpaceDimension >
void SetOffset1( const itk::Vector< TScalarType, SpaceDimension > &,
                 cl_float &, SpaceDimensionToType< SpaceDimension > )
{}

template< class TScalarType, unsigned int SpaceDimension >
void SetOffset2( const itk::Vector< TScalarType, SpaceDimension > &,
                 cl_float2 &, SpaceDimensionToType< SpaceDimension > )
{}

template< class TScalarType, unsigned int SpaceDimension >
void SetOffset3( const itk::Vector< TScalarType, SpaceDimension > &,
                 cl_float3 &, SpaceDimensionToType< SpaceDimension > )
{}

template< class TScalarType >
void SetOffset1( const itk::Vector< TScalarType, 1 > & offset,
                 cl_float & ocloffset, SpaceDimensionToType< 1 > )
{
  ocloffset = offset[0];
}

template< class TScalarType >
void SetOffset2( const itk::Vector< TScalarType, 2 > & offset,
                 cl_float2 & ocloffset, SpaceDimensionToType< 2 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 2; i++ )
  {
    ocloffset.s[id++] = offset[i];
  }
}

template< class TScalarType >
void SetOffset3( const itk::Vector< TScalarType, 3 > & offset,
                 cl_float3 & ocloffset, SpaceDimensionToType< 3 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 3; i++ )
  {
    ocloffset.s[id++] = offset[i];
  }
  ocloffset.s[3] = 0.0;
}

//----------------------------------------------------------------------------
// Matrix
template< class TScalarType,
          unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension >
void SetMatrix1( const itk::Matrix< TScalarType, OutputSpaceDimension, InputSpaceDimension > &,
                 cl_float &,
                 SpaceDimensionToType< OutputSpaceDimension >,
                 SpaceDimensionToType< InputSpaceDimension > )
{}

template< class TScalarType,
          unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension >
void SetMatrix2( const itk::Matrix< TScalarType, OutputSpaceDimension, InputSpaceDimension > &,
                 cl_float4 &,
                 SpaceDimensionToType< OutputSpaceDimension >,
                 SpaceDimensionToType< InputSpaceDimension > )
{}

template< class TScalarType,
          unsigned int OutputSpaceDimension, unsigned int InputSpaceDimension >
void SetMatrix3( const itk::Matrix< TScalarType, OutputSpaceDimension, InputSpaceDimension > &,
                 cl_float16 &,
                 SpaceDimensionToType< OutputSpaceDimension >,
                 SpaceDimensionToType< InputSpaceDimension > )
{}

template< class TScalarType >
void SetMatrix1( const itk::Matrix< TScalarType, 1, 1 > & matrix,
                 cl_float & oclmatrix, SpaceDimensionToType< 1 >, SpaceDimensionToType< 1 > )
{
  oclmatrix = matrix[0][0];
}

template< class TScalarType >
void SetMatrix2( const itk::Matrix< TScalarType, 2, 2 > & matrix,
                 cl_float4 & oclmatrix, SpaceDimensionToType< 2 >, SpaceDimensionToType< 2 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 2; i++ )
  {
    for ( unsigned int j = 0; j < 2; j++ )
    {
      oclmatrix.s[id++] = matrix[i][j];
    }
  }
}

template< class TScalarType >
void SetMatrix3( const itk::Matrix< TScalarType, 3, 3 > & matrix,
                 cl_float16 & oclmatrix, SpaceDimensionToType< 3 >, SpaceDimensionToType< 3 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 3; i++ )
  {
    for ( unsigned int j = 0; j < 3; j++ )
    {
      oclmatrix.s[id++] = matrix[i][j];
    }
  }
  for ( unsigned int i = 9; i < 16; i++ )
  {
    oclmatrix.s[i] = 0.0;
  }
}
} // end of ITKGPUMatrixOffsetTransformBase namespace

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
GPUMatrixOffsetTransformBase< TScalarType, NInputDimensions, NOutputDimensions >
::GPUMatrixOffsetTransformBase()
{
  // Add GPUMatrixOffsetTransformBase source
  const std::string sourcePath(
    GPUMatrixOffsetTransformBaseKernel::GetOpenCLSource() );

  m_Sources.push_back( sourcePath );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag( CL_MEM_READ_ONLY );

  using namespace ITKGPUMatrixOffsetTransformBase;
  const unsigned int OutputDimension = OutputSpaceDimension;

  switch ( OutputDimension )
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUMatrixOffsetTransformBase1D ) );
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUMatrixOffsetTransformBase2D ) );
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUMatrixOffsetTransformBase3D ) );
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
GPUDataManager::Pointer GPUMatrixOffsetTransformBase< TScalarType, NInputDimensions, NOutputDimensions >
::GetParametersDataManager() const
{
  using namespace ITKGPUMatrixOffsetTransformBase;
  const unsigned int InputDimension  = InputSpaceDimension;
  const unsigned int OutputDimension = OutputSpaceDimension;
  const SpaceDimensionToType< InputSpaceDimension >  idim = {};
  const SpaceDimensionToType< OutputSpaceDimension > odim = {};

  switch ( OutputDimension )
  {
    case 1:
    {
      GPUMatrixOffsetTransformBase1D transformBase;
      SetMatrix1< ScalarType >( GetCPUMatrix(), transformBase.matrix, odim, idim );
      SetOffset1< ScalarType >( GetCPUOffset(), transformBase.offset, odim );
      SetMatrix1< ScalarType >( GetCPUInverseMatrix(), transformBase.inverse_matrix, idim, odim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &transformBase );
    }
    break;
    case 2:
    {
      GPUMatrixOffsetTransformBase2D transformBase;
      SetMatrix2< ScalarType >( GetCPUMatrix(), transformBase.matrix, odim, idim );
      SetOffset2< ScalarType >( GetCPUOffset(), transformBase.offset, odim );
      SetMatrix2< ScalarType >( GetCPUInverseMatrix(), transformBase.inverse_matrix, idim, odim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &transformBase );
    }
    break;
    case 3:
    {
      GPUMatrixOffsetTransformBase3D transformBase;
      SetMatrix3< ScalarType >( GetCPUMatrix(), transformBase.matrix, odim, idim );
      SetOffset3< ScalarType >( GetCPUOffset(), transformBase.offset, odim );
      SetMatrix3< ScalarType >( GetCPUInverseMatrix(), transformBase.inverse_matrix, idim, odim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &transformBase );
    }
    break;
    default:
      break;
  }

  this->m_ParametersDataManager->SetGPUDirtyFlag( true );
  this->m_ParametersDataManager->UpdateGPUBuffer();

  return this->m_ParametersDataManager;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions >
bool GPUMatrixOffsetTransformBase< TScalarType, NInputDimensions, NOutputDimensions >
::GetSourceCode( std::string & _source ) const
{
  if ( !m_SourcesLoaded )
  {
    return false;
  }

  // Create the final source code
  std::ostringstream source;
  // Add other sources
  for ( std::size_t i = 0; i < m_Sources.size(); i++ )
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

} // end namespace itk

#endif /* __itkGPUMatrixOffsetTransformBase_hxx */
