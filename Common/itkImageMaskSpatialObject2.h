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
  Date:      $Date: 2008-05-28 10:45:42 +0200 (Wed, 28 May 2008) $
  Version:   $Revision: 1636 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkImageMaskSpatialObject2_h
#define __itkImageMaskSpatialObject2_h

#include "itkImageSpatialObject2.h"
#include "itkImageSliceConstIteratorWithIndex.h"

namespace itk
{

/** \class ImageMaskSpatialObject2
 * \brief Implementation of an image mask as spatial object.
 *
 * This class fixes a bug in the ITK. The ITK has implemented
 * the ImageSpatialObject with a wrong conversion between physical
 * coordinates and image coordinates. This class solves that.
 *
 */

template< unsigned int TDimension = 3 >
class ImageMaskSpatialObject2 :
  public ImageSpatialObject2< TDimension, unsigned char >
{

public:

  typedef ImageMaskSpatialObject2< TDimension > Self;
  typedef ImageSpatialObject2< TDimension >     Superclass;
  typedef SmartPointer< Self >                  Pointer;
  typedef SmartPointer< const Self >            ConstPointer;

  typedef typename Superclass::ScalarType      ScalarType;
  typedef typename Superclass::PixelType       PixelType;
  typedef typename Superclass::ImageType       ImageType;
  typedef typename Superclass::ImagePointer    ImagePointer;
  typedef typename Superclass::IndexType       IndexType;
  typedef typename Superclass::SizeType        SizeType;
  typedef typename Superclass::RegionType      RegionType;
  typedef typename Superclass::TransformType   TransformType;
  typedef typename Superclass::PointType       PointType;
  typedef typename Superclass::BoundingBoxType BoundingBoxType;

  typedef itk::ImageSliceConstIteratorWithIndex< ImageType >
    SliceIteratorType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageMaskSpatialObject2, ImageSpatialObject2 );

  /** Returns true if the point is inside, false otherwise. */
  bool IsInside( const PointType & point,
    unsigned int depth, char * name ) const;

  /** Test whether a point is inside or outside the object
  *  For computational speed purposes, it is faster if the method does not
  *  check the name of the class and the current depth */
  virtual bool IsInside( const PointType & point ) const;

  /** Compute axis aligned bounding box from the image mask. The bounding box
   * is returned as an image region. Each call to this function will recompute
   * the region. This function is useful in cases, where you may have a mask image
   * resulting from say a segmentation and you want to get the smallest box
   * region that encapsulates the mask image. Currently this is done only for 3D
   * volumes. */
  RegionType GetAxisAlignedBoundingBoxRegion() const;

  /** Compute the boundaries of the image mask spatial object. */
  bool ComputeLocalBoundingBox() const;

  /** Helper function for GetAxisAlignedBoundingBoxRegion()
   * and ComputeLocalBoundingBox().
   */
  void ComputeLocalBoundingBoxIndexAndSize(
    IndexType & index, SizeType & size ) const;

protected:

  ImageMaskSpatialObject2( const Self & ); // purposely not implemented
  void operator=( const Self & );          // purposely not implemented

  ImageMaskSpatialObject2();
  virtual ~ImageMaskSpatialObject2();

  void PrintSelf( std::ostream & os, Indent indent ) const;

};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageMaskSpatialObject2.hxx"
#endif

#endif //__itkImageMaskSpatialObject2_h
