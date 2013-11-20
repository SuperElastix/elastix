/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkCyclicGridScheduleComputer_h__
#define __itkCyclicGridScheduleComputer_h__

#include "itkImageBase.h"
#include "itkTransform.h"
#include "itkGridScheduleComputer.h"

namespace itk
{

/**
 * \class CyclicGridScheduleComputer
 *
 * \brief This class computes all information about the B-spline grid.
 *
 * This class computes all information about the B-spline grid
 * given the image information and the desired grid spacing. It differs from
 * the GridScheduleComputer in how the nodes are placed in the last dimension.
 *
 * \ingroup Transforms
 */

template < typename TTransformScalarType, unsigned int VImageDimension >
class ITK_EXPORT CyclicGridScheduleComputer
  : public GridScheduleComputer< TTransformScalarType, VImageDimension >
{
public:

  /** Standard class typedefs. */
  typedef CyclicGridScheduleComputer          Self;
  typedef GridScheduleComputer<
      TTransformScalarType, VImageDimension>  Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( CyclicGridScheduleComputer, GridScheduleComputer );

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
  typedef std::vector< RegionType >             VectorRegionType;
  typedef std::vector< GridSpacingFactorType >  VectorGridSpacingFactorType;

  /** Typedefs for the initial transform. */
  typedef Transform<
    TransformScalarType,
    itkGetStaticConstMacro( Dimension ),
    itkGetStaticConstMacro( Dimension ) >       TransformType;
  typedef typename TransformType::Pointer       TransformPointer;
  typedef typename TransformType::ConstPointer  TransformConstPointer;

  /** Compute the B-spline grid. */
  virtual void ComputeBSplineGrid( void );

protected:

  /** The constructor. */
  CyclicGridScheduleComputer();

  /** The destructor. */
  virtual ~CyclicGridScheduleComputer() {};

private:

  CyclicGridScheduleComputer( const Self& );  // purposely not implemented
  void operator=( const Self& );        // purposely not implemented

}; // end class CyclicGridScheduleComputer

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCyclicGridScheduleComputer.hxx"
#endif

#endif // end #ifndef __itkCyclicGridScheduleComputer_h__
