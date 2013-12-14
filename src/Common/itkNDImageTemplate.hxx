/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __itkNDImageTemplate_hxx
#define __itkNDImageTemplate_hxx

#include "itkNDImageTemplate.h"

namespace itk
{

/** Constructor */
template< class TPixel, unsigned int VDimension >
NDImageTemplate< TPixel, VDimension >::NDImageTemplate()
{
  this->m_Image  = 0;
  this->m_Writer = 0;
  this->m_Reader = 0;
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetRegions( SizeType size )
{
  this->m_Image->SetRegions(
    ConvertToStaticArray< SizeType, SizeTypeD >::DO( size ) );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetRequestedRegion( DataObject * data )
{
  this->m_Image->SetRequestedRegion( data );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::Allocate( void )
{
  this->m_Image->Allocate();
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::Initialize( void )
{
  this->m_Image->Initialize();
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::FillBuffer( const TPixel & value )
{
  this->m_Image->FillBuffer( value );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetPixel( const IndexType & index, const TPixel & value )
{
  this->m_Image->SetPixel(
    ConvertToStaticArray< IndexType, IndexTypeD >::DO( index ),
    value );
}


template< class TPixel, unsigned int VDimension >
const TPixel &
NDImageTemplate< TPixel, VDimension >::GetPixel( const IndexType & index ) const
{
  return this->m_Image->GetPixel(
    ConvertToStaticArray< IndexType, IndexTypeD >::DO( index ) );
}


template< class TPixel, unsigned int VDimension >
TPixel &
NDImageTemplate< TPixel, VDimension >::GetPixel( const IndexType & index )
{
  return this->m_Image->GetPixel(
    ConvertToStaticArray< IndexType, IndexTypeD >::DO( index ) );
}


template< class TPixel, unsigned int VDimension >
TPixel *
NDImageTemplate< TPixel, VDimension >::GetBufferPointer()
{
  return this->m_Image->GetBufferPointer();
}


template< class TPixel, unsigned int VDimension >
const TPixel *
NDImageTemplate< TPixel, VDimension >::GetBufferPointer() const
{
  return this->m_Image->GetBufferPointer();
}


template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::PixelContainer
* NDImageTemplate< TPixel, VDimension >::
GetPixelContainer()
{
  return this->m_Image->GetPixelContainer();
}

template< class TPixel, unsigned int VDimension >
const typename NDImageTemplate< TPixel, VDimension >::PixelContainer
* NDImageTemplate< TPixel, VDimension >::
GetPixelContainer() const
{
  return this->m_Image->GetPixelContainer();
}

template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetPixelContainer( PixelContainer * container )
{
  this->m_Image->SetPixelContainer( container );
}


template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::AccessorType
NDImageTemplate< TPixel, VDimension >::GetPixelAccessor( void )
{
  return this->m_Image->GetPixelAccessor();
}


template< class TPixel, unsigned int VDimension >
const typename NDImageTemplate< TPixel, VDimension >::AccessorType
NDImageTemplate< TPixel, VDimension >::GetPixelAccessor( void ) const
{
  return this->m_Image->GetPixelAccessor();
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetSpacing( const SpacingType & spacing )
{
  this->m_Image->SetSpacing(
    ConvertToStaticArray< SpacingType, SpacingTypeD >::DO( spacing ) );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetOrigin( const PointType & origin )
{
  this->m_Image->SetOrigin(
    ConvertToStaticArray< PointType, PointTypeD >::DO( origin ) );
}


template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::SpacingType
NDImageTemplate< TPixel, VDimension >::GetSpacing( void )
{
  return ConvertToDynamicArray< SpacingTypeD, SpacingType >::DO(
    this->m_Image->GetSpacing() );
}


template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::PointType
NDImageTemplate< TPixel, VDimension >::GetOrigin( void )
{
  return ConvertToDynamicArray< PointTypeD, PointType >::DO(
    this->m_Image->GetOrigin() );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::CopyInformation( const DataObject * data )
{
  this->m_Image->CopyInformation( data );
}


template< class TPixel, unsigned int VDimension >
const typename NDImageTemplate< TPixel, VDimension >::OffsetValueType
* NDImageTemplate< TPixel, VDimension >::
GetOffsetTable() const
{
  return this->m_Image->GetOffsetTable();
}

template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::OffsetValueType
NDImageTemplate< TPixel, VDimension >::ComputeOffset( const IndexType & ind ) const
{
  return this->m_Image->ComputeOffset(
    ConvertToStaticArray< IndexType, IndexTypeD >::DO( ind ) );
}


template< class TPixel, unsigned int VDimension >
typename NDImageTemplate< TPixel, VDimension >::IndexType
NDImageTemplate< TPixel, VDimension >::ComputeIndex( OffsetValueType offset ) const
{
  return ConvertToDynamicArray< IndexTypeD, IndexType >::DO(
    this->m_Image->ComputeIndex( offset ) );
}


template< class TPixel, unsigned int VDimension >
unsigned int
NDImageTemplate< TPixel, VDimension >::ImageDimension( void )
{
  return this->m_Image->GetImageDimension();
}


template< class TPixel, unsigned int VDimension >
unsigned int
NDImageTemplate< TPixel, VDimension >::GetImageDimension( void )
{
  return this->m_Image->GetImageDimension();
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::Write( void )
{
  if( this->m_Writer )
  {
    this->m_Writer->SetInput( this->m_Image );
    this->m_Writer->Write();
  }
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::Read( void )
{
  if( this->m_Reader )
  {
    this->m_Reader->Update();
    this->m_Image = this->m_Reader->GetOutput();
  }
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::CreateNewImage( void )
{
  this->m_Image = ImageType::New();
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetImageIOWriter( ImageIOBase * _arg )
{
  if( !( this->m_Writer ) )
  {
    this->m_Writer = WriterType::New();
  }
  this->m_Writer->SetImageIO( _arg );
}


template< class TPixel, unsigned int VDimension >
ImageIOBase *
NDImageTemplate< TPixel, VDimension >::GetImageIOWriter( void )
{
  if( this->m_Writer )
  {
    return this->m_Writer->GetImageIO();
  }
  else
  {
    return 0;
  }
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetImageIOReader( ImageIOBase * _arg )
{
  if( !( this->m_Reader ) )
  {
    this->m_Reader = ReaderType::New();
  }
  this->m_Reader->SetImageIO( _arg );
}


template< class TPixel, unsigned int VDimension >
ImageIOBase *
NDImageTemplate< TPixel, VDimension >::GetImageIOReader( void )
{
  if( this->m_Reader )
  {
    return this->m_Reader->GetImageIO();
  }
  else
  {
    return 0;
  }
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetOutputFileName( const char * name )
{
  if( !( this->m_Writer ) )
  {
    this->m_Writer = WriterType::New();
  }
  this->m_Writer->SetFileName( name );
}


template< class TPixel, unsigned int VDimension >
void
NDImageTemplate< TPixel, VDimension >::SetInputFileName( const char * name )
{
  if( !( this->m_Reader ) )
  {
    this->m_Reader = ReaderType::New();
  }
  this->m_Reader->SetFileName( name );
}


template< class TPixel, unsigned int VDimension >
const char *
NDImageTemplate< TPixel, VDimension >::GetOutputFileName( void )
{
  if( this->m_Writer )
  {
    return this->m_Writer->GetFileName();
  }
  else
  {
    return "";
  }
}


template< class TPixel, unsigned int VDimension >
const char *
NDImageTemplate< TPixel, VDimension >::GetInputFileName( void )
{
  if( this->m_Reader )
  {
    return this->m_Reader->GetFileName().c_str();
  }
  else
  {
    return "";
  }
}


} // end namespace itk

#endif // end #ifndef __itkNDImageTemplate_hxx
