/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUInterpolateImageFunction_hxx
#define __itkGPUInterpolateImageFunction_hxx

#include "itkGPUInterpolateImageFunction.h"
#include "itkOCLOstreamSupport.h"

// begin of unnamed namespace
namespace
{
typedef struct {
  cl_uint start_index;
  cl_uint end_index;
  cl_float start_continuous_index;
  cl_float end_continuous_index;
} GPUImageFunction1D;

typedef struct {
  cl_uint2 start_index;
  cl_uint2 end_index;
  cl_float2 start_continuous_index;
  cl_float2 end_continuous_index;
} GPUImageFunction2D;

typedef struct {
  cl_uint4 start_index;
  cl_uint4 end_index;
  cl_float4 start_continuous_index;
  cl_float4 end_continuous_index;
} GPUImageFunction3D;

//------------------------------------------------------------------------------
template< unsigned int ImageDimension >
struct ImageDimensionToType {};

// IndexType
template< class ImageType, unsigned int ImageDimension >
void SetIndex( const typename ImageType::IndexType,
               cl_uint &, ImageDimensionToType< ImageDimension > )
{}

template< class ImageType, unsigned int ImageDimension >
void SetIndex( const typename ImageType::IndexType,
               cl_uint2 &, ImageDimensionToType< ImageDimension > )
{}

template< class ImageType, unsigned int ImageDimension >
void SetIndex( const typename ImageType::IndexType,
               cl_uint4 &, ImageDimensionToType< ImageDimension > )
{}

// ContinuousIndexType
template< class ImageType, class TCoordRep, unsigned int ImageDimension >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, ImageDimension >,
                         cl_float &, ImageDimensionToType< ImageDimension > )
{}

template< class ImageType, class TCoordRep, unsigned int ImageDimension >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, ImageDimension >,
                         cl_float2 &, ImageDimensionToType< ImageDimension > )
{}

template< class ImageType, class TCoordRep, unsigned int ImageDimension >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, ImageDimension >,
                         cl_float4 &, ImageDimensionToType< ImageDimension > )
{}

// IndexType
template< class ImageType >
void SetIndex( const typename ImageType::IndexType index,
               cl_uint & oclindex, ImageDimensionToType< 1 > )
{
  oclindex = index[0];
}

template< class ImageType >
void SetIndex( const typename ImageType::IndexType index,
               cl_uint2 & oclindex, ImageDimensionToType< 2 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 2; i++ )
  {
    oclindex.s[id++] = index[i];
  }
}

template< class ImageType >
void SetIndex( const typename ImageType::IndexType index,
               cl_uint4 & oclindex, ImageDimensionToType< 3 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 3; i++ )
  {
    oclindex.s[id++] = index[i];
  }
  oclindex.s[3] = 0;
}

// ContinuousIndexType
template< class ImageType, class TCoordRep >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, 1 > index,
                         cl_float & oclindex, ImageDimensionToType< 1 > )
{
  oclindex = index[0];
}

template< class ImageType, class TCoordRep >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, 2 > index,
                         cl_float2 & oclindex, ImageDimensionToType< 2 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 2; i++ )
  {
    oclindex.s[id++] = index[i];
  }
}

template< class ImageType, class TCoordRep >
void SetContinuousIndex( const itk::ContinuousIndex< TCoordRep, 3 > index,
                         cl_float4 & oclindex, ImageDimensionToType< 3 > )
{
  unsigned int id = 0;

  for ( unsigned int i = 0; i < 3; i++ )
  {
    oclindex.s[id++] = index[i];
  }
  oclindex.s[3] = 0.0;
}
} // end of unnamed namespace

//------------------------------------------------------------------------------
namespace itk
{
template< class TInputImage, class TCoordRep, class TParentImageFilter >
GPUInterpolateImageFunction< TInputImage, TCoordRep, TParentImageFilter >
::GPUInterpolateImageFunction()
{
  const unsigned int ImageDim = InputImageType::ImageDimension;

  this->m_ParametersDataManager->Initialize();
  this->m_ParametersDataManager->SetBufferFlag( CL_MEM_READ_ONLY );

  switch ( ImageDim )
  {
    case 1:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUImageFunction1D ) );
      break;
    case 2:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUImageFunction2D ) );
      break;
    case 3:
      this->m_ParametersDataManager->SetBufferSize( sizeof( GPUImageFunction3D ) );
      break;
    default:
      break;
  }

  this->m_ParametersDataManager->Allocate();
}

//------------------------------------------------------------------------------
template< class TInputImage, class TCoordRep, class TParentImageFilter >
GPUDataManager::Pointer GPUInterpolateImageFunction< TInputImage, TCoordRep, TParentImageFilter >
::GetParametersDataManager() const
{
  const unsigned int                                ImageDim = InputImageType::ImageDimension;
  const ImageDimensionToType< InputImageDimension > idim = {};

  switch ( ImageDim )
  {
    case 1:
    {
      GPUImageFunction1D imageFunction;

      SetIndex< InputImageType >( this->m_StartIndex, imageFunction.start_index, idim );
      SetIndex< InputImageType >( this->m_EndIndex, imageFunction.end_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_StartContinuousIndex,
                                                          imageFunction.start_continuous_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_EndContinuousIndex,
                                                          imageFunction.end_continuous_index, idim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &imageFunction );
    }
    break;
    case 2:
    {
      GPUImageFunction2D imageFunction;

      SetIndex< InputImageType >( this->m_StartIndex, imageFunction.start_index, idim );
      SetIndex< InputImageType >( this->m_EndIndex, imageFunction.end_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_StartContinuousIndex,
                                                          imageFunction.start_continuous_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_EndContinuousIndex,
                                                          imageFunction.end_continuous_index, idim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &imageFunction );
    }
    break;
    case 3:
    {
      GPUImageFunction3D imageFunction;

      SetIndex< InputImageType >( this->m_StartIndex, imageFunction.start_index, idim );
      SetIndex< InputImageType >( this->m_EndIndex, imageFunction.end_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_StartContinuousIndex,
                                                          imageFunction.start_continuous_index, idim );
      SetContinuousIndex< InputImageType, CoordRepType >( this->m_EndContinuousIndex,
                                                          imageFunction.end_continuous_index, idim );
      this->m_ParametersDataManager->SetCPUBufferPointer( &imageFunction );
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
template< class TInputImage, class TCoordRep, class TParentImageFilter >
void GPUInterpolateImageFunction< TInputImage, TCoordRep, TParentImageFilter >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}
} // end namespace itk

#endif /* __itkGPUInterpolateImageFunction_hxx */
