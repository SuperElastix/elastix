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

/** This file is a slightly modified version of an ITK file.
 * Original ITK copyright message: */
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date: 2008-05-29 12:02:25 +0200 (Thu, 29 May 2008) $
  Version:   $Revision: 1641 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __ImageMaskSpatialObject2_hxx
#define __ImageMaskSpatialObject2_hxx

#include "itkImageMaskSpatialObject2.h"
#include "vnl/vnl_math.h"

#include "itkImageRegionConstIteratorWithIndex.h"

namespace itk
{

/** Constructor */
template< unsigned int TDimension >
ImageMaskSpatialObject2< TDimension >
::ImageMaskSpatialObject2()
{
  this->SetTypeName( "ImageMaskSpatialObject2" );
  this->ComputeBoundingBox();
}


/** Destructor */
template< unsigned int TDimension >
ImageMaskSpatialObject2< TDimension >
::~ImageMaskSpatialObject2()
{}

/** Test whether a point is inside or outside the object
*  For computational speed purposes, it is faster if the method does not
*  check the name of the class and the current depth */
template< unsigned int TDimension >
bool
ImageMaskSpatialObject2< TDimension >
::IsInside( const PointType & point ) const
{
  if( !this->GetBounds()->IsInside( point ) )
  {
    return false;
  }
  if( !this->SetInternalInverseTransformToWorldToIndexTransform() )
  {
    return false;
  }

  PointType p = this->GetInternalInverseTransform()->TransformPoint( point );

  IndexType index;
  for( unsigned int i = 0; i < TDimension; i++ )
  {
    //index[i] = static_cast<int>( p[i] ); // changed by stefan
    index[ i ] = static_cast< int >( Math::Round< double >( p[ i ] ) );
  }

  const bool insideBuffer
    = this->GetImage()->GetBufferedRegion().IsInside( index );

  if( !insideBuffer )
  {
    return false;
  }

  const bool insideMask
    = ( this->GetImage()->GetPixel( index ) != NumericTraits< PixelType >::ZeroValue() );

  return insideMask;

}


/** Return true if the given point is inside the image */
template< unsigned int TDimension >
bool
ImageMaskSpatialObject2< TDimension >
::IsInside( const PointType & point, unsigned int depth, char * name ) const
{
  if( name == NULL )
  {
    if( IsInside( point ) )
    {
      return true;
    }
  }
  else if( strstr( typeid( Self ).name(), name ) )
  {
    if( IsInside( point ) )
    {
      return true;
    }
  }
  return SpatialObject< TDimension >::IsInside( point, depth, name );
}


// is this one correct? (stefan).
template< unsigned int TDimension >
typename ImageMaskSpatialObject2< TDimension >::RegionType
ImageMaskSpatialObject2< TDimension >
::GetAxisAlignedBoundingBoxRegion() const
{
  /** Compute index and size of the bounding box. */
  IndexType index;
  SizeType  size;
  this->ComputeLocalBoundingBoxIndexAndSize( index, size );

  /** Define and return region. */
  RegionType region;
  region.SetIndex( index );
  region.SetSize( size );

  return region;

}  // end GetAxisAlignedBoundingBoxRegion()


template< unsigned int TDimension >
void
ImageMaskSpatialObject2< TDimension >
::ComputeLocalBoundingBoxIndexAndSize(
  IndexType & index, SizeType & size ) const
{
  // We will use a slice iterator to iterate through slices orthogonal
  // to each of the axis of the image to find the bounding box. Each
  // slice iterator iterates from the outermost slice towards the image
  // center till it finds a mask pixel. For a 3D image, there will be six
  // slice iterators, iterating from the periphery inwards till the bounds
  // along each axes are found. The slice iterators save time and avoid
  // having to walk the whole image. Since we are using slice iterators,
  // we will implement this only for 3D images.

  ImagePointer image        = this->GetImage();
  PixelType    outsideValue = NumericTraits< PixelType >::Zero;

  // Initialize index and size in case image only consists of background values.
  for( unsigned int axis = 0; axis < ImageType::ImageDimension; ++axis )
  {
    index[ axis ] = 0;
    size[ axis ] = 0;
  }

  /** For 3D a smart implementation existed in the ITK already. */
  if( ImageType::ImageDimension == 3 )
  {
    for( unsigned int axis = 0; axis < ImageType::ImageDimension; axis++ )
    {
      // Two slice iterators along each axis...
      // Find the orthogonal planes for the slices
      unsigned int i, j;
      unsigned int direction[ 2 ];
      for( i = 0, j = 0; i < 3; ++i )
      {
        if( i != axis )
        {
          direction[ j ] = i;
          j++;
        }
      }

      // Create the forward iterator to find lower bound
      SliceIteratorType fit( image,  image->GetRequestedRegion() );
      fit.SetFirstDirection(  direction[ 1 ] );
      fit.SetSecondDirection( direction[ 0 ] );

      fit.GoToBegin();
      while( !fit.IsAtEnd() )
      {
        while( !fit.IsAtEndOfSlice() )
        {
          while( !fit.IsAtEndOfLine() )
          {
            if( fit.Get() !=  outsideValue )
            {
              index[ axis ] = fit.GetIndex()[ axis ];
              fit.GoToReverseBegin(); // skip to the end
              break;
            }
            ++fit;
          }
          fit.NextLine();
        }
        fit.NextSlice();
      }

      // Create the reverse iterator to find upper bound
      SliceIteratorType rit( image,  image->GetRequestedRegion() );
      rit.SetFirstDirection(  direction[ 1 ] );
      rit.SetSecondDirection( direction[ 0 ] );

      rit.GoToReverseBegin();
      while( !rit.IsAtReverseEnd() )
      {
        while( !rit.IsAtReverseEndOfSlice() )
        {
          while( !rit.IsAtReverseEndOfLine() )
          {
            if( rit.Get() !=  outsideValue )
            {
              //size[ axis ] = rit.GetIndex()[ axis ] - index[ axis ]; // changed by Marius
              size[ axis ] = rit.GetIndex()[ axis ] - index[ axis ] + 1;
              rit.GoToBegin(); //Skip to reverse end
              break;
            }
            --rit;
          }
          rit.PreviousLine();
        }
        rit.PreviousSlice();
      }
    }
  }
  // We added a naive implementation for images of dimension other than 3
  else
  {
    typedef ImageRegionConstIteratorWithIndex< ImageType > IteratorType;
    IteratorType it( image, image->GetRequestedRegion() );
    it.GoToBegin();
    IndexType endindex;

    for( unsigned int i = 0; i < ImageType::ImageDimension; ++i )
    {
      // modified by stefan; old (commented) implemenation assumed zero start index
      index[ i ] = image->GetRequestedRegion().GetIndex( i ) + image->GetRequestedRegion().GetSize( i ) - 1;
      //index[ i ] = image->GetRequestedRegion().GetSize( i );
      endindex[ i ] = image->GetRequestedRegion().GetIndex( i );
      //size[ i ]  = image->GetRequestedRegion().GetIndex( i );
    }

    while( !it.IsAtEnd() )
    {
      if( it.Get() != outsideValue )
      {
        IndexType tmpIndex = it.GetIndex();
        for( unsigned int i = 0; i < ImageType::ImageDimension; ++i )
        {
          index[ i ] = index[ i ] < tmpIndex[ i ] ? index[ i ] : tmpIndex[ i ];
          //size[ i ] = static_cast<long>( size[ i ] )  > tmpIndex[ i ] ? size[ i ]  : tmpIndex[ i ];
          endindex[ i ] = endindex[ i ] > tmpIndex[ i ] ? endindex[ i ]  : tmpIndex[ i ];
        }
      }
      ++it;
    }

    for( unsigned int i = 0; i < ImageType::ImageDimension; ++i )
    {
      //size[ i ] = size[ i ] - index[ i ] + 1;
      size[ i ] = endindex[ i ] - index[ i ] + 1;
    }
  } // end else

} // end ComputeLocalBoundingBoxIndexAndSize()


/** Compute the bounds of the image */
template< unsigned int TDimension >
bool
ImageMaskSpatialObject2< TDimension >
::ComputeLocalBoundingBox() const
{
  if( this->GetBoundingBoxChildrenName().empty()
    || strstr( typeid( Self ).name(),
    this->GetBoundingBoxChildrenName().c_str() ) )
  {
    /** Compute index and size of the bounding box. */
    IndexType indexLow;
    SizeType  size;
    this->ComputeLocalBoundingBoxIndexAndSize( indexLow, size );

    /** Convert to points, which are NOT physical points! */
    PointType pointLow, pointHigh;
    for( unsigned int i = 0; i < ImageType::ImageDimension; ++i )
    {
      pointLow[ i ]  = indexLow[ i ];
      pointHigh[ i ] = indexLow[ i ] + size[ i ] - 1;
    }

    /** Compute the bounding box. */
    typename BoundingBoxType::Pointer bb = BoundingBoxType::New();
    bb->SetMinimum( pointLow );
    bb->SetMaximum( pointHigh );
    typedef typename BoundingBoxType::PointsContainer PointsContainerType;
    const PointsContainerType * corners = bb->GetCorners();

    /** Take into account indextoworld transform: SK: itk implementation was buggy */
    typename PointsContainerType::Pointer cornersWorld = PointsContainerType::New();
    cornersWorld->Reserve( corners->Size() );

    typename PointsContainerType::const_iterator itC = corners->begin();
    typename PointsContainerType::iterator itCW      = cornersWorld->begin();
    while( itC != corners->end() )
    {
      PointType transformedPoint = this->GetIndexToWorldTransform()->TransformPoint( *itC );
      *itCW = transformedPoint;
      itCW++;
      itC++;
    }
    const_cast< BoundingBoxType * >( this->GetBounds() )->SetPoints( cornersWorld );
    const_cast< BoundingBoxType * >( this->GetBounds() )->ComputeBoundingBox();

    return true;
  }

  return false;

} // end ComputeLocalBoundingBox()


/** Print the object */
template< unsigned int TDimension >
void
ImageMaskSpatialObject2< TDimension >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
}


} // end namespace itk

#endif //__ImageMaskSpatialObject2_hxx
