/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUAffineTransform_hxx
#define __itkGPUAffineTransform_hxx

#include "itkGPUAffineTransform.h"
#include "itkGPUMatrixOffsetTransformBase.h"
#include <iomanip>

// begin of unnamed namespace
namespace
{
typedef struct {
  cl_float Matrix;
  cl_float Offset;
  cl_float InverseMatrix;
} GPUMatrixOffsetTransformBase1D;

typedef struct {
  cl_float4 Matrix;
  cl_float2 Offset;
  cl_float4 InverseMatrix;
} GPUMatrixOffsetTransformBase2D;

typedef struct {
  cl_float16 Matrix;        // OpenCL does not have float9
  cl_float4 Offset;
  cl_float16 InverseMatrix; // OpenCL does not have float9
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
                 cl_float4 &, SpaceDimensionToType< SpaceDimension > )
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
                 cl_float4 & ocloffset, SpaceDimensionToType< 3 > )
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
} // end of unnamed namespace

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUAffineTransform< TScalarType, NDimensions, TParentImageFilter >::GPUAffineTransform() :
  Superclass( ParametersDimension )
{
  // Add GPUMatrixOffsetTransformBase header
  const std::string sourcePath0(
    GPUMatrixOffsetTransformBaseHeaderKernel::GetOpenCLSource() );
  m_Sources.push_back( sourcePath0 );

  // Add GPUMatrixOffsetTransformBase source
  const std::string sourcePath1(
    GPUMatrixOffsetTransformBaseKernel::GetOpenCLSource() );
  m_Sources.push_back( sourcePath1 );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings

  const unsigned int InputDimension  = InputSpaceDimension;
  const unsigned int OutputDimension = OutputSpaceDimension;

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag( CL_MEM_READ_ONLY );

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
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
GPUDataManager::Pointer GPUAffineTransform< TScalarType, NDimensions, TParentImageFilter >
::GetParametersDataManager() const
{
  const unsigned int InputDimension  = InputSpaceDimension;
  const unsigned int OutputDimension = OutputSpaceDimension;
  const SpaceDimensionToType< InputSpaceDimension >  idim = {};
  const SpaceDimensionToType< OutputSpaceDimension > odim = {};

  switch ( OutputDimension )
  {
    case 1:
    {
      GPUMatrixOffsetTransformBase1D transformBase;
      SetMatrix1< ScalarType >( this->GetMatrix(), transformBase.Matrix, odim, idim );
      SetOffset1< ScalarType >( this->GetOffset(), transformBase.Offset, odim );
      SetMatrix1< ScalarType >( this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &transformBase );
    }
    break;
    case 2:
    {
      GPUMatrixOffsetTransformBase2D transformBase;
      SetMatrix2< ScalarType >( this->GetMatrix(), transformBase.Matrix, odim, idim );
      SetOffset2< ScalarType >( this->GetOffset(), transformBase.Offset, odim );
      SetMatrix2< ScalarType >( this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &transformBase );
    }
    break;
    case 3:
    {
      GPUMatrixOffsetTransformBase3D transformBase;
      SetMatrix3< ScalarType >( this->GetMatrix(), transformBase.Matrix, odim, idim );
      SetOffset3< ScalarType >( this->GetOffset(), transformBase.Offset, odim );
      SetMatrix3< ScalarType >( this->GetInverseMatrix(), transformBase.InverseMatrix, idim, odim );
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
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
bool GPUAffineTransform< TScalarType, NDimensions, TParentImageFilter >
::GetSourceCode( std::string & _source ) const
{
  if ( !m_SourcesLoaded )
  {
    return false;
  }

  // Create the final source code
  std::ostringstream source;
  // Add other sources
  for ( unsigned int i = 0; i < m_Sources.size(); i++ )
  {
    source << m_Sources[i] << std::endl;
  }
  _source = source.str();
  return true;
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions, class TParentImageFilter >
void GPUAffineTransform< TScalarType, NDimensions, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}

} // end namespace itk

#endif /* __itkGPUAffineTransform_hxx */
