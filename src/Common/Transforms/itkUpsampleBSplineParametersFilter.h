/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkUpsampleBSplineParametersFilter_h
#define __itkUpsampleBSplineParametersFilter_h

#include "itkObject.h"
#include "itkArray.h"

namespace itk
{

/** \class UpsampleBSplineParametersFilter
 *
 * \brief Convenience class for upsampling a B-spline coefficient image.
 *
 * The UpsampleBSplineParametersFilter class is a class that takes as input
 * the B-spline parameters. It's purpose is to compute new B-spline parameters
 * on a denser grid. Therefore, the user needs to supply the old B-spline grid
 * (region, spacing, origin, direction), and the required B-spline grid.
 *
 */

template< class TArray, class TImage >
class UpsampleBSplineParametersFilter :
  public Object
{
public:

  /** Standard class typedefs. */
  typedef UpsampleBSplineParametersFilter Self;
  typedef Object                          Superclass;
  typedef SmartPointer< Self >            Pointer;
  typedef SmartPointer< const Self >      ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( UpsampleBSplineParametersFilter, Object );

  /** Typedefs. */
  typedef TArray                            ArrayType;
  typedef typename ArrayType::ValueType     ValueType;
  typedef TImage                            ImageType;
  typedef typename ImageType::Pointer       ImagePointer;
  typedef typename ImageType::PixelType     PixelType;
  typedef typename ImageType::SpacingType   SpacingType;
  typedef typename ImageType::PointType     OriginType;
  typedef typename ImageType::DirectionType DirectionType;
  typedef typename ImageType::RegionType    RegionType;

  /** Dimension of the fixed image. */
  itkStaticConstMacro( Dimension, unsigned int, ImageType::ImageDimension );

  /** Set the origin of the current grid. */
  itkSetMacro( CurrentGridOrigin, OriginType );

  /** Set the spacing of the current grid. */
  itkSetMacro( CurrentGridSpacing, SpacingType );

  /** Set the direction of the current grid. */
  itkSetMacro( CurrentGridDirection, DirectionType );

  /** Set the region of the current grid. */
  itkSetMacro( CurrentGridRegion, RegionType );

  /** Set the origin of the required grid. */
  itkSetMacro( RequiredGridOrigin, OriginType );

  /** Set the spacing of the required grid. */
  itkSetMacro( RequiredGridSpacing, SpacingType );

  /** Set the direction of the required grid. */
  itkSetMacro( RequiredGridDirection, DirectionType );

  /** Set the region of the required grid. */
  itkSetMacro( RequiredGridRegion, RegionType );

  /** Set the B-spline order. */
  itkSetMacro( BSplineOrder, unsigned int );

  /** Compute the output parameter array. */
  virtual void UpsampleParameters( const ArrayType & param_in,
    ArrayType & param_out );

protected:

  /** Constructor. */
  UpsampleBSplineParametersFilter();

  /** Destructor. */
  ~UpsampleBSplineParametersFilter() {}

  /** PrintSelf. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Function that checks if upsampling is required. */
  virtual bool DoUpsampling( void );

private:

  UpsampleBSplineParametersFilter( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

  /** Private member variables. */
  OriginType    m_CurrentGridOrigin;
  SpacingType   m_CurrentGridSpacing;
  DirectionType m_CurrentGridDirection;
  RegionType    m_CurrentGridRegion;
  OriginType    m_RequiredGridOrigin;
  SpacingType   m_RequiredGridSpacing;
  DirectionType m_RequiredGridDirection;
  RegionType    m_RequiredGridRegion;
  unsigned int  m_BSplineOrder;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkUpsampleBSplineParametersFilter.hxx"
#endif

#endif // end #ifndef __itkUpsampleBSplineParametersFilter_h
