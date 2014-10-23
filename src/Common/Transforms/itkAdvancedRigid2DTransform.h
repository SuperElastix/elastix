/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedRigid2DTransform.h,v $
  Language:  C++
  Date:      $Date: 2009-01-14 18:39:05 $
  Version:   $Revision: 1.22 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedRigid2DTransform_h
#define __itkAdvancedRigid2DTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkExceptionObject.h"

namespace itk
{

/** \class AdvancedRigid2DTransform
 * \brief AdvancedRigid2DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rigid transformation in 2D space.
 * The transform is specified as a rotation around a arbitrary center
 * and is followed by a translation.
 *
 * The parameters for this transform can be set either using
 * individual Set methods or in serialized form using
 * SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 3 elements
 * ordered as follows:
 * p[0] = angle
 * p[1] = x component of the translation
 * p[2] = y component of the translation
 *
 * The serialization of the fixed parameters is an array of 2 elements
 * ordered as follows:
 * p[0] = x coordinate of the center
 * p[1] = y coordinate of the center
 *
 * Access methods for the center, translation and underlying matrix
 * offset vectors are documented in the superclass AdvancedMatrixOffsetTransformBase.
 *
 * \sa Transform
 * \sa AdvancedMatrixOffsetTransformBase
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
// Data type for scalars (float or double)
class AdvancedRigid2DTransform :
  public AdvancedMatrixOffsetTransformBase< TScalarType, 2, 2 >      // Dimensions of input and output spaces
{
public:

  /** Standard class typedefs. */
  typedef AdvancedRigid2DTransform                               Self;
  typedef AdvancedMatrixOffsetTransformBase< TScalarType, 2, 2 > Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedRigid2DTransform, AdvancedMatrixOffsetTransformBase );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Dimension of the space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, 2 );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, 2 );
  itkStaticConstMacro( ParametersDimension, unsigned int, 3 );

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType JacobianType;

  /// Standard matrix type for this class
  typedef typename Superclass::MatrixType MatrixType;

  /// Standard vector type for this class
  typedef typename Superclass::OffsetType OffsetType;

  /// Standard vector type for this class
  typedef typename Superclass::InputVectorType  InputVectorType;
  typedef typename Superclass::OutputVectorType OutputVectorType;

  /// Standard covariant vector type for this class
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;

  /// Standard vnl_vector type for this class
  typedef typename Superclass::InputVnlVectorType  InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;

  /// Standard coordinate point type for this class
  typedef typename Superclass::InputPointType  InputPointType;
  typedef typename Superclass::OutputPointType OutputPointType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /**
   * Set the rotation Matrix of a Rigid2D Transform
   *
   * This method sets the 2x2 matrix representing the rotation
   * in the transform.  The Matrix is expected to be orthogonal
   * with a certain tolerance.
   *
   * \warning This method will throw an exception is the matrix
   * provided as argument is not orthogonal.
   *
   * \sa AdvancedMatrixOffsetTransformBase::SetMatrix()
   */
  virtual void SetMatrix( const MatrixType & matrix );

  /**
   * Set/Get the rotation matrix. These methods are old and are
   * retained for backward compatibility. Instead, use SetMatrix()
   * GetMatrix().
   */
  virtual void SetRotationMatrix( const MatrixType & matrix )
  { this->SetMatrix( matrix ); }
  const MatrixType & GetRotationMatrix() const
  { return this->GetMatrix(); }

  /**
   * Compose the transformation with a translation
   *
   * This method modifies self to include a translation of the
   * origin.  The translation is precomposed with self if pre is
   * true, and postcomposed otherwise.
   */
  void Translate( const OffsetType & offset, bool pre = false );

  /**
   * Back transform by an rigid transformation.
   *
   * The BackTransform() methods are slated to be removed from ITK.
   * Instead, please use GetInverse() or CloneInverseTo() to generate
   * an inverse transform and  then perform the transform using that
   * inverted transform.
   */
  inline InputPointType      BackTransform( const OutputPointType  & point ) const;

  inline InputVectorType     BackTransform( const OutputVectorType & vector ) const;

  inline InputVnlVectorType  BackTransform( const OutputVnlVectorType & vector ) const;

  inline InputCovariantVectorType BackTransform(
    const OutputCovariantVectorType & vector ) const;

  /** Set/Get the angle of rotation in radians */
  void SetAngle( TScalarType angle );

  itkGetConstReferenceMacro( Angle, TScalarType );

  /** Set the angle of rotation in degrees. */
  void SetAngleInDegrees( TScalarType angle );

  /** Set/Get the angle of rotation in radians. These methods
   * are old and are retained for backward compatibility.
   * Instead, use SetAngle() and GetAngle(). */
  void SetRotation( TScalarType angle )
  { this->SetAngle( angle ); }
  virtual const TScalarType & GetRotation() const
  { return m_Angle; }

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.
   * There are 3 parameters. The first one represents the
   * angle of rotation in radians and the last two represents the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::SetParameters()
   * \sa Transform::SetFixedParameters() */
  void SetParameters( const ParametersType & parameters );

  /** Get the parameters that uniquely define the transform
   * This is typically used by optimizers.
   * There are 3 parameters. The first one represents the
   * angle or rotation in radians and the last two represents the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::GetParameters()
   * \sa Transform::GetFixedParameters() */
  const ParametersType & GetParameters( void ) const;

  /** This method computes the Jacobian matrix of the transformation
   * at a given input point.
   *
   * \sa Transform::GetJacobian() */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /**
   * This method creates and returns a new AdvancedRigid2DTransform object
   * which is the inverse of self.
   */
  void CloneInverseTo( Pointer & newinverse ) const;

  /**
   * This method creates and returns a new AdvancedRigid2DTransform object
   * which has the same parameters.
   */
  void CloneTo( Pointer & clone ) const;

  /** Reset the parameters to create and identity transform. */
  virtual void SetIdentity( void );

protected:

  AdvancedRigid2DTransform();
  AdvancedRigid2DTransform( unsigned int parametersDimension );
  AdvancedRigid2DTransform( unsigned int outputSpaceDimension, unsigned int parametersDimension );

  ~AdvancedRigid2DTransform();

  /**
    * Print contents of an AdvancedRigid2DTransform
    */
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Compute the matrix from angle. This is used in Set methods
   * to update the underlying matrix whenever a transform parameter
   * is changed.
   * Also update the m_JacobianOfSpatialJacobian. */
  virtual void ComputeMatrix( void );

  /** Compute the angle from the matrix. This is used to compute
   * transform parameters from a given matrix. This is used in
   * AdvancedMatrixOffsetTransformBase::Compose() and
   * AdvancedMatrixOffsetTransformBase::GetInverse(). */
  virtual void ComputeMatrixParameters( void );

  /** Update angle without recomputation of other internal variables. */
  void SetVarAngle( TScalarType angle )
  { m_Angle = angle; }

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

private:

  AdvancedRigid2DTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );           // purposely not implemented

  TScalarType m_Angle;

};

// Back transform a point
template< class TScalarType >
inline
typename AdvancedRigid2DTransform< TScalarType >::InputPointType
AdvancedRigid2DTransform< TScalarType >::BackTransform( const OutputPointType & point ) const
{
  itkWarningMacro(
      << "BackTransform(): This method is slated to be removed from ITK.  Instead, please use GetInverse() to generate an inverse transform and then perform the transform using that inverted transform."
    );
  return this->GetInverseMatrix() * ( point - this->GetOffset() );
}


// Back transform a vector
template< class TScalarType >
inline
typename AdvancedRigid2DTransform< TScalarType >::InputVectorType
AdvancedRigid2DTransform< TScalarType >
::BackTransform( const OutputVectorType & vect ) const
{
  itkWarningMacro(
      << "BackTransform(): This method is slated to be removed from ITK.  Instead, please use GetInverse() to generate an inverse transform and then perform the transform using that inverted transform."
    );
  return this->GetInverseMatrix() * vect;
}


// Back transform a vnl_vector
template< class TScalarType >
inline
typename AdvancedRigid2DTransform< TScalarType >::InputVnlVectorType
AdvancedRigid2DTransform< TScalarType >
::BackTransform( const OutputVnlVectorType & vect ) const
{
  itkWarningMacro(
      << "BackTransform(): This method is slated to be removed from ITK.  Instead, please use GetInverse() to generate an inverse transform and then perform the transform using that inverted transform."
    );
  return this->GetInverseMatrix() * vect;
}


// Back Transform a CovariantVector
template< class TScalarType >
inline
typename AdvancedRigid2DTransform< TScalarType >::InputCovariantVectorType
AdvancedRigid2DTransform< TScalarType >
::BackTransform( const OutputCovariantVectorType & vect ) const
{
  itkWarningMacro(
      << "BackTransform(): This method is slated to be removed from ITK.  Instead, please use GetInverse() to generate an inverse transform and then perform the transform using that inverted transform."
    );
  return this->GetMatrix() * vect;
}


}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedRigid2DTransform.hxx"
#endif

#endif /* __itkAdvancedRigid2DTransform_h */
