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
#ifndef __itkImageSpatialObject2_h
#define __itkImageSpatialObject2_h

#include "itkImage.h"
#include "itkExceptionObject.h"
#include "itkSpatialObject.h"
#include "itkInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

namespace itk
{

/** \class ImageSpatialObject2
 * \brief Implementation of an image as spatial object.
 *
 * This class fixes a bug in the ITK. The ITK has implemented
 * the ImageSpatialObject with a wrong conversion between physical
 * coordinates and image coordinates. This class solves that.
 *
 * \sa SpatialObject CompositeSpatialObject
 */

template< unsigned int TDimension = 3,
class TPixelType                  = unsigned char
>
class ImageSpatialObject2 :
  public SpatialObject< TDimension >
{

public:

  typedef double                                        ScalarType;
  typedef ImageSpatialObject2< TDimension, TPixelType > Self;
  typedef SpatialObject< TDimension >                   Superclass;
  typedef SmartPointer< Self >                          Pointer;
  typedef SmartPointer< const Self >                    ConstPointer;

  typedef TPixelType                            PixelType;
  typedef Image< PixelType, TDimension >        ImageType;
  typedef typename ImageType::ConstPointer      ImagePointer;
  typedef typename ImageType::IndexType         IndexType;
  typedef typename ImageType::SizeType          SizeType;
  typedef typename ImageType::RegionType        RegionType;
  typedef typename Superclass::TransformType    TransformType;
  typedef typename Superclass::PointType        PointType;
  typedef typename Superclass::BoundingBoxType  BoundingBoxType;
  typedef InterpolateImageFunction< ImageType > InterpolatorType;

  typedef NearestNeighborInterpolateImageFunction< ImageType >
    NNInterpolatorType;

  typedef VectorContainer< unsigned long, PointType > PointContainerType;
  typedef typename PointContainerType::Pointer        PointContainerPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ImageSpatialObject2, SpatialObject );

  /** Set the image. */
  void SetImage( const ImageType * image );

  /** Get a pointer to the image currently attached to the object. */
  const ImageType * GetImage( void ) const;

  /** Return true if the object is evaluable at the requested point,
   *  and else otherwise. */
  bool IsEvaluableAt( const PointType & point,
    unsigned int depth = 0, char * name = NULL ) const;

  /** Returns the value of the image at the requested point.
   *  If the point is not inside the object, then an exception is thrown.
   * \sa ExceptionObject */
  bool ValueAt( const PointType & point, double & value,
    unsigned int depth = 0, char * name = NULL ) const;

  /** Returns true if the point is inside, false otherwise. */
  bool IsInside( const PointType & point,
    unsigned int depth, char * name ) const;

  /** Test whether a point is inside or outside the object
   *  For computational speed purposes, it is faster if the method does not
   *  check the name of the class and the current depth */
  bool IsInside( const PointType & point ) const;

  /** Compute the boundaries of the image spatial object. */
  bool ComputeLocalBoundingBox() const;

  /** Returns the latest modified time of the object and its component. */
  unsigned long GetMTime( void ) const;

  /** Set the slice position */
  void SetSlicePosition( unsigned int dimension, int position );

  /** Get the slice position */
  int GetSlicePosition( unsigned int dimension )
  { return m_SlicePosition[ dimension ]; }

  const char * GetPixelType()
  {
    return m_PixelType.c_str();
  }


  /** Set/Get the interpolator */
  void SetInterpolator( InterpolatorType * interpolator );

  itkGetObjectMacro( Interpolator, InterpolatorType );

protected:

  ImageSpatialObject2( const Self & ); // purposely not implemented
  void operator=( const Self & );      // purposely not implemented

  ImagePointer m_Image;

  ImageSpatialObject2();
  virtual ~ImageSpatialObject2();

  void PrintSelf( std::ostream & os, Indent indent ) const;

  int *       m_SlicePosition;
  std::string m_PixelType;

  typename InterpolatorType::Pointer m_Interpolator;
};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkImageSpatialObject2.hxx"
#endif

#endif //__itkImageSpatialObject2_h
