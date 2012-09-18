/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUTranslationTransformBase_hxx
#define __itkGPUTranslationTransformBase_hxx

#include "itkGPUTranslationTransformBase.h"
#include <iomanip>

// begin of ITKGPUTranslationTransformBase namespace
namespace ITKGPUTranslationTransformBase
{
typedef struct {
  cl_float offset;
} GPUTranslationTransformBase1D;

typedef struct {
  cl_float2 offset;
} GPUTranslationTransformBase2D;

typedef struct {
  cl_float3 offset;
} GPUTranslationTransformBase3D;

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
} // end of ITKGPUTranslationTransformBase namespace

//------------------------------------------------------------------------------
namespace itk
{
template< class TScalarType, unsigned int NDimensions >
GPUTranslationTransformBase< TScalarType, NDimensions >
::GPUTranslationTransformBase()
{
  // Add GPUTranslationTransformBase source
  const std::string sourcePath(
    GPUTranslationTransformBaseKernel::GetOpenCLSource() );

  m_Sources.push_back( sourcePath );

  m_SourcesLoaded = true; // we set it to true, sources are loaded from strings

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag( CL_MEM_READ_ONLY );

  using namespace ITKGPUTranslationTransformBase;
  const unsigned int Dimension = SpaceDimension;

  switch ( Dimension )
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUTranslationTransformBase1D ) );
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUTranslationTransformBase2D ) );
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUTranslationTransformBase3D ) );
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
}

//------------------------------------------------------------------------------
template< class TScalarType, unsigned int NDimensions >
GPUDataManager::Pointer GPUTranslationTransformBase< TScalarType, NDimensions >
::GetParametersDataManager() const
{
  using namespace ITKGPUTranslationTransformBase;
  const SpaceDimensionToType< SpaceDimension > dim = {};
  const unsigned int Dimension = SpaceDimension;

  switch ( Dimension )
  {
    case 1:
    {
      GPUTranslationTransformBase1D translationBase;
      SetOffset1< ScalarType >( GetCPUOffset(), translationBase.offset, dim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &translationBase );
    }
    break;
    case 2:
    {
      GPUTranslationTransformBase2D translationBase;
      SetOffset2< ScalarType >( GetCPUOffset(), translationBase.offset, dim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &translationBase );
    }
    break;
    case 3:
    {
      GPUTranslationTransformBase3D translationBase;
      SetOffset3< ScalarType >( GetCPUOffset(), translationBase.offset, dim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &translationBase );
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
template< class TScalarType, unsigned int NDimensions >
bool GPUTranslationTransformBase< TScalarType, NDimensions >
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

#endif /* __itkGPUTranslationTransformBase_hxx */
