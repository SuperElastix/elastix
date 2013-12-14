/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedEuler3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.14 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedEuler3DTransform_h
#define __itkAdvancedEuler3DTransform_h

#include <iostream>
#include "itkAdvancedRigid3DTransform.h"

namespace itk
{

/** \class AdvancedEuler3DTransform
 *
 * \brief AdvancedEuler3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation and translation to the space given 3 euler
 * angles and a 3D translation. Rotation is about a user specified center.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 6 elements.
 * The first 3 represents three euler angle of rotation respectively about
 * the X, Y and Z axis. The last 3 parameters defines the translation in each
 * dimension.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
// Data type for scalars (float or double)
class ITK_EXPORT AdvancedEuler3DTransform :
  public         AdvancedRigid3DTransform< TScalarType >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedEuler3DTransform                Self;
  typedef AdvancedRigid3DTransform< TScalarType > Superclass;
  typedef SmartPointer< Self >                    Pointer;
  typedef SmartPointer< const Self >              ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedEuler3DTransform, AdvancedRigid3DTransform );

  /** Dimension of the space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( InputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( ParametersDimension, unsigned int, 6 );

  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::MatrixType                MatrixType;
  typedef typename Superclass::InverseMatrixType         InverseMatrixType;
  typedef typename Superclass::CenterType                CenterType;
  typedef typename Superclass::TranslationType           TranslationType;
  typedef typename Superclass::OffsetType                OffsetType;
  typedef typename Superclass::ScalarType                AngleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /** Set/Get the transformation from a container of parameters
   * This is typically used by optimizers.  There are 6 parameters. The first
   * three represent the angles to rotate around the coordinate axis, and the
   * last three represents the offset. */
  void SetParameters( const ParametersType & parameters );

  const ParametersType & GetParameters( void ) const;

  /** Set the rotational part of the transform. */
  void SetRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ );

  itkGetConstMacro( AngleX, ScalarType );
  itkGetConstMacro( AngleY, ScalarType );
  itkGetConstMacro( AngleZ, ScalarType );

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Set/Get the order of the computation. Default ZXY */
  itkSetMacro( ComputeZYX, bool );
  itkGetConstMacro( ComputeZYX, bool );

  virtual void SetIdentity( void );

protected:

  AdvancedEuler3DTransform();
  AdvancedEuler3DTransform( const MatrixType & matrix,
    const OutputPointType & offset );
  AdvancedEuler3DTransform( unsigned int paramsSpaceDims );

  ~AdvancedEuler3DTransform(){}

  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Set values of angles directly without recomputing other parameters. */
  void SetVarRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ );

  /** Compute the components of the rotation matrix in the superclass. */
  void ComputeMatrix( void );

  void ComputeMatrixParameters( void );

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

private:

  AdvancedEuler3DTransform( const Self & ); //purposely not implemented
  void operator=( const Self & );           //purposely not implemented

  ScalarType m_AngleX;
  ScalarType m_AngleY;
  ScalarType m_AngleZ;
  bool       m_ComputeZYX;

};

//class AdvancedEuler3DTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedEuler3DTransform.hxx"
#endif

#endif /* __itkAdvancedEuler3DTransform_h */
