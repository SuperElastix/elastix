/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkSingleValuedPointSetToPointSetMetric.h,v $
  Language:  C++
  Date:      $Date: 2009-01-26 21:45:56 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkSingleValuedPointSetToPointSetMetric_h
#define __itkSingleValuedPointSetToPointSetMetric_h

#include "itkImageBase.h"
#include "itkAdvancedTransform.h"
#include "itkSingleValuedCostFunction.h"
#include "itkExceptionObject.h"
#include "itkSpatialObject.h"
#include "itkPointSet.h"


namespace itk
{

/** \class SingleValuedPointSetToPointSetMetric
 * \brief Computes similarity between two point sets.
 *
 * This Class is templated over the type of the two point-sets. It
 * expects a Transform to be plugged in. This particular
 * class is the base class for a hierarchy of point-set to point-set metrics.
 *
 * This class computes a value that measures the similarity between the fixed point-set
 * and the transformed moving point-set.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedPointSet, class TMovingPointSet>
class ITK_EXPORT SingleValuedPointSetToPointSetMetric
  : public SingleValuedCostFunction
{
public:

  /** Standard class typedefs. */
  typedef SingleValuedPointSetToPointSetMetric  Self;
  typedef SingleValuedCostFunction              Superclass;
  typedef SmartPointer<Self>                    Pointer;
  typedef SmartPointer<const Self>              ConstPointer;

  /** Type used for representing point components  */
  typedef Superclass::ParametersValueType CoordinateRepresentationType;

  /** Run-time type information (and related methods). */
  itkTypeMacro( SingleValuedPointSetToPointSetMetric, SingleValuedCostFunction );

  /** Typedefs. */
  typedef TFixedPointSet                             FixedPointSetType;
  typedef typename FixedPointSetType::PixelType      FixedPointSetPixelType;
  typedef typename FixedPointSetType::ConstPointer   FixedPointSetConstPointer;
  typedef TMovingPointSet                            MovingPointSetType;
  typedef typename MovingPointSetType::PixelType     MovingPointSetPixelType;
  typedef typename MovingPointSetType::ConstPointer  MovingPointSetConstPointer;
  typedef typename FixedPointSetType::PointsContainer::ConstIterator    PointIterator;
  typedef typename FixedPointSetType::PointDataContainer::ConstIterator PointDataIterator;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro( FixedPointSetDimension, unsigned int,
    TFixedPointSet::PointDimension );
  itkStaticConstMacro( MovingPointSetDimension, unsigned int,
    TMovingPointSet::PointDimension );

  /**  More typedefs. */
  typedef AdvancedTransform< CoordinateRepresentationType,
    itkGetStaticConstMacro( FixedPointSetDimension ),
    itkGetStaticConstMacro( MovingPointSetDimension ) > TransformType;
  typedef typename TransformType::Pointer               TransformPointer;
  typedef typename TransformType::InputPointType        InputPointType;
  typedef typename TransformType::OutputPointType       OutputPointType;
  typedef typename TransformType::ParametersType        TransformParametersType;
  typedef typename TransformType::JacobianType          TransformJacobianType;

  typedef SpatialObject<
      itkGetStaticConstMacro( FixedPointSetDimension )> FixedImageMaskType;
  typedef typename FixedImageMaskType::Pointer          FixedImageMaskPointer;
  typedef typename FixedImageMaskType::ConstPointer     FixedImageMaskConstPointer;
  typedef SpatialObject<
    itkGetStaticConstMacro( MovingPointSetDimension )>  MovingImageMaskType;
  typedef typename MovingImageMaskType::Pointer         MovingImageMaskPointer;
  typedef typename MovingImageMaskType::ConstPointer    MovingImageMaskConstPointer;

  /**  Type of the measure. */
  typedef Superclass::MeasureType                       MeasureType;
  typedef Superclass::DerivativeType                    DerivativeType;
  typedef typename DerivativeType::ValueType            DerivativeValueType;
  typedef Superclass::ParametersType                    ParametersType;

  /** Typedefs for support of sparse Jacobians and compact support of transformations. */
  typedef typename TransformType::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Connect the fixed pointset.  */
  itkSetConstObjectMacro( FixedPointSet, FixedPointSetType );

  /** Get the fixed pointset. */
  itkGetConstObjectMacro( FixedPointSet, FixedPointSetType );

  /** Connect the moving pointset.  */
  itkSetConstObjectMacro( MovingPointSet, MovingPointSetType );

  /** Get the moving pointset. */
  itkGetConstObjectMacro( MovingPointSet, MovingPointSetType );

  /** Connect the Transform. */
  itkSetObjectMacro( Transform, TransformType );

  /** Get a pointer to the Transform.  */
  itkGetConstObjectMacro( Transform, TransformType );

  /** Set the parameters defining the Transform. */
  void SetTransformParameters( const ParametersType & parameters ) const;

  /** Return the number of parameters required by the transform. */
  unsigned int GetNumberOfParameters( void ) const
  { return this->m_Transform->GetNumberOfParameters(); }

  /** Initialize the Metric by making sure that all the components are
   *  present and plugged together correctly.
   */
  virtual void Initialize( void ) throw ( ExceptionObject );

  /** Set the fixed mask. */
  // \todo: currently not used
  itkSetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /** Get the fixed mask. */
  itkGetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /** Set the moving mask. */
  itkSetConstObjectMacro( MovingImageMask, MovingImageMaskType );

  /** Get the moving mask. */
  itkGetConstObjectMacro( MovingImageMask, MovingImageMaskType );

protected:
  SingleValuedPointSetToPointSetMetric();
  virtual ~SingleValuedPointSetToPointSetMetric() {};

  /** PrintSelf. */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Member variables. */
  FixedPointSetConstPointer   m_FixedPointSet;
  MovingPointSetConstPointer  m_MovingPointSet;
  FixedImageMaskConstPointer  m_FixedImageMask;
  MovingImageMaskConstPointer m_MovingImageMask;
  mutable TransformPointer    m_Transform;

  mutable unsigned int        m_NumberOfPointsCounted;

private:
  SingleValuedPointSetToPointSetMetric(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

}; // end class SingleValuedPointSetToPointSetMetric

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSingleValuedPointSetToPointSetMetric.txx"
#endif

#endif
