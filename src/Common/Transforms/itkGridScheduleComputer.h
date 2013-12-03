/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGridScheduleComputer_H__
#define __itkGridScheduleComputer_H__

#include "itkObject.h"
#include "itkImageBase.h"
#include "itkTransform.h"

namespace itk
{

/**
 * \class GridScheduleComputer
 * \brief This class computes all information about the B-spline grid,
 * given the image information and the desired grid spacing.
 *
 * NB: the Direction Cosines of the B-spline grid are set identical
 * to the user-supplied ImageDirection.
 *
 * \ingroup Transforms
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
class GridScheduleComputer
  : public Object
{
public:

  /** Standard class typedefs. */
  typedef GridScheduleComputer                Self;
  typedef Object                              Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( GridScheduleComputer, Object );

  /** Dimension of the domain space. */
  itkStaticConstMacro( Dimension, unsigned int, VImageDimension );

  /** Typedef's. */
  typedef TTransformScalarType                  TransformScalarType;
  typedef ImageBase<
    itkGetStaticConstMacro( Dimension ) >       ImageBaseType;
  typedef typename ImageBaseType::PointType     PointType;
  typedef typename ImageBaseType::PointType     OriginType;
  typedef typename ImageBaseType::SpacingType   SpacingType;
  typedef typename ImageBaseType::DirectionType DirectionType;
  typedef typename ImageBaseType::SizeType      SizeType;
  typedef typename ImageBaseType::SizeValueType SizeValueType;
  typedef typename ImageBaseType::RegionType    RegionType;
  typedef SpacingType                           GridSpacingFactorType;
  typedef std::vector< OriginType >             VectorOriginType;
  typedef std::vector< SpacingType >            VectorSpacingType;
  typedef std::vector< DirectionType >          VectorDirectionType;
  typedef std::vector< RegionType >             VectorRegionType;
  typedef std::vector< GridSpacingFactorType >  VectorGridSpacingFactorType;

  /** Typedefs for the initial transform. */
  typedef Transform<
    TransformScalarType,
    itkGetStaticConstMacro( Dimension ),
    itkGetStaticConstMacro( Dimension ) >       TransformType;
  typedef typename TransformType::Pointer       TransformPointer;
  typedef typename TransformType::ConstPointer  TransformConstPointer;

  /** Set the ImageOrigin. */
  itkSetMacro( ImageOrigin, OriginType );

  /** Get the ImageOrigin. */
  itkGetConstMacro( ImageOrigin, OriginType );

  /** Set the ImageSpacing. */
  itkSetMacro( ImageSpacing, SpacingType );

  /** Get the ImageSpacing. */
  itkGetConstMacro( ImageSpacing, SpacingType );

  /** Set the ImageDirection. */
  itkSetMacro( ImageDirection, DirectionType );

  /** Get the ImageDirection. */
  itkGetConstMacro( ImageDirection, DirectionType );

  /** Set the ImageRegion. */
  itkSetMacro( ImageRegion, RegionType );

  /** Get the ImageRegion. */
  itkGetConstMacro( ImageRegion, RegionType );

  /** Set the B-spline order. */
  itkSetClampMacro( BSplineOrder, unsigned int, 0, 5 );

  /** Get the B-spline order. */
  itkGetConstMacro( BSplineOrder, unsigned int );

  /** Set the final grid spacing. */
  itkSetMacro( FinalGridSpacing, SpacingType );

  /** Get the final grid spacing. */
  itkGetConstMacro( FinalGridSpacing, SpacingType );

  /** Set a default grid spacing schedule. */
  virtual void SetDefaultSchedule(
    unsigned int levels,
    double upsamplingFactor );

  /** Set a grid spacing schedule. */
  virtual void SetSchedule(
    const VectorGridSpacingFactorType & schedule );

  /** Get the grid spacing schedule. */
  virtual void GetSchedule( VectorGridSpacingFactorType & schedule ) const;

  /** Set an initial Transform. Only set one if composition is used. */
  itkSetConstObjectMacro( InitialTransform, TransformType );

  /** Compute the B-spline grid. */
  virtual void ComputeBSplineGrid( void );

  /** Get the B-spline grid at some level. */
  virtual void GetBSplineGrid( unsigned int level,
    RegionType & gridRegion,
    SpacingType & gridSpacing,
    OriginType & gridOrigin,
    DirectionType & gridDirection );

protected:

  /** The constructor. */
  GridScheduleComputer();

  /** The destructor. */
  virtual ~GridScheduleComputer() {};

  /** Declare member variables, needed for B-spline grid. */
  VectorSpacingType     m_GridSpacings;
  VectorOriginType      m_GridOrigins;
  VectorDirectionType   m_GridDirections;
  VectorRegionType      m_GridRegions;
  TransformConstPointer m_InitialTransform;
  VectorGridSpacingFactorType m_GridSpacingFactors;

  /** PrintSelf. */
  void PrintSelf( std::ostream& os, Indent indent ) const;

  /** Get number of levels. */
  itkGetConstMacro( NumberOfLevels, unsigned int );

  /** Function to apply the initial transform, if it exists. */
  virtual void ApplyInitialTransform(
    OriginType & imageOrigin,
    SpacingType & imageSpacing,
    DirectionType & imageDirection,
    SpacingType & finalGridSpacing ) const;

private:

  GridScheduleComputer( const Self& );  // purposely not implemented
  void operator=( const Self& );        // purposely not implemented

  /** Declare member variables, needed in functions. */
  OriginType            m_ImageOrigin;
  SpacingType           m_ImageSpacing;
  RegionType            m_ImageRegion;
  DirectionType         m_ImageDirection;
  unsigned int          m_BSplineOrder;
  unsigned int          m_NumberOfLevels;
  SpacingType           m_FinalGridSpacing;

  /** Clamp the upsampling factor. */
  itkSetClampMacro( UpsamplingFactor, float, 1.0, NumericTraits<float>::max() );

  /** Declare member variables, needed internally. */
  float                 m_UpsamplingFactor;

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGridScheduleComputer.hxx"
#endif

#endif // end #ifndef __itkGridScheduleComputer_H__
