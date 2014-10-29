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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedSimilarity2DTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-06-07 16:06:32 $
  Version:   $Revision: 1.11 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedSimilarity2DTransform_h
#define __itkAdvancedSimilarity2DTransform_h

#include <iostream>
#include "itkAdvancedRigid2DTransform.h"

namespace itk
{

/** \brief AdvancedSimilarity2DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a homogenous scale and rigid transform in
 * 2D space. The transform is specified as a scale and rotation around
 * a arbitrary center and is followed by a translation.
 * given one angle for rotation, a homogeneous scale and a 2D offset for translation.
 *
 * The parameters for this transform can be set either using
 * individual Set methods or in serialized form using
 * SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 3 elements
 * ordered as follows:
 * p[0] = scale
 * p[1] = angle
 * p[2] = x component of the translation
 * p[3] = y component of the translation
 *
 * The serialization of the fixed parameters is an array of 2 elements
 * ordered as follows:
 * p[0] = x coordinate of the center
 * p[1] = y coordinate of the center
 *
 * Access methods for the center, translation and underlying matrix
 * offset vectors are documented in the superclass MatrixOffsetTransformBase.
 *
 * Access methods for the angle are documented in superclass Rigid2DTransform.
 *
 * \sa Transform
 * \sa MatrixOffsetTransformBase
 * \sa Rigid2DTransform
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
// Data type for scalars (float or double)
class ITK_EXPORT AdvancedSimilarity2DTransform :
  public         AdvancedRigid2DTransform< TScalarType >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedSimilarity2DTransform           Self;
  typedef AdvancedRigid2DTransform< TScalarType > Superclass;
  typedef SmartPointer< Self >                    Pointer;
  typedef SmartPointer< const Self >              ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedSimilarity2DTransform, AdvancedRigid2DTransform );

  /** Dimension of parameters. */
  itkStaticConstMacro( SpaceDimension,           unsigned int, 2 );
  itkStaticConstMacro( InputSpaceDimension,      unsigned int, 2 );
  itkStaticConstMacro( OutputSpaceDimension,     unsigned int, 2 );
  itkStaticConstMacro( ParametersDimension,      unsigned int, 4 );

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;
  typedef          TScalarType            ScaleType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;

  /** Jacobian type. */
  typedef typename Superclass::JacobianType JacobianType;

  /** Offset type. */
  typedef typename Superclass::OffsetType OffsetType;

  /** Matrix type. */
  typedef typename Superclass::MatrixType MatrixType;

  /** Point type. */
  typedef typename Superclass::InputPointType  InputPointType;
  typedef typename Superclass::OutputPointType OutputPointType;

  /** Vector type. */
  typedef typename Superclass::InputVectorType  InputVectorType;
  typedef typename Superclass::OutputVectorType OutputVectorType;

  /** CovariantVector type. */
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;

  /** VnlVector type. */
  typedef typename Superclass::InputVnlVectorType  InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType OutputVnlVectorType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  /** Set the Scale part of the transform. */
  void SetScale( ScaleType scale );

  itkGetConstReferenceMacro( Scale, ScaleType );

  /** Set the transformation from a container of parameters
    * This is typically used by optimizers.
    * There are 4 parameters. The first one represents the
    * scale, the second represents the angle of rotation
    * and the last two represent the translation.
    * The center of rotation is fixed.
    *
    * \sa Transform::SetParameters()
    * \sa Transform::SetFixedParameters() */
  void SetParameters( const ParametersType & parameters );

  /** Get the parameters that uniquely define the transform
   * This is typically used by optimizers.
   * There are 4 parameters. The first one represents the
   * scale, the second represents the angle of rotation,
   * and the last two represent the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::GetParameters()
   * \sa Transform::GetFixedParameters() */
  const ParametersType & GetParameters( void ) const;

  /** This method computes the Jacobian matrix of the transformation
   * at a given input point.
   *
   * \sa Transform::GetJacobian() */

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  /** Set the transformation to an identity. */
  virtual void SetIdentity( void );

  /**
   * This method creates and returns a new AdvancedSimilarity2DTransform object
   * which is the inverse of self.
   **/
  void CloneInverseTo( Pointer & newinverse ) const;

  /**
   * This method creates and returns a new AdvancedSimilarity2DTransform object
   * which has the same parameters.
   **/
  void CloneTo( Pointer & clone ) const;

  /**
   * Set the rotation Matrix of a Similarity 2D Transform
   *
   * This method sets the 2x2 matrix representing a similarity
   * transform.  The Matrix is expected to be a valid
   * similarity transform with a certain tolerance.
   *
   * \warning This method will throw an exception if the matrix
   * provided as argument is not valid.
   *
   * \sa MatrixOffsetTransformBase::SetMatrix()
   *
   **/
  virtual void SetMatrix( const MatrixType & matrix );

protected:

  AdvancedSimilarity2DTransform();
  AdvancedSimilarity2DTransform( unsigned int spaceDimension,
    unsigned int parametersDimension );

  ~AdvancedSimilarity2DTransform(){}
  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Compute matrix from angle and scale. This is used in Set methods
   * to update the underlying matrix whenever a transform parameter
   * is changed. */
  virtual void ComputeMatrix( void );

  /** Compute the angle and scale from the matrix. This is used to compute
   * transform parameters from a given matrix. This is used in
   * MatrixOffsetTransformBase::Compose() and
   * MatrixOffsetTransformBase::GetInverse(). */
  virtual void ComputeMatrixParameters( void );

  /** Set the scale without updating underlying variables. */
  void SetVarScale( ScaleType scale )
  { m_Scale = scale; }

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

private:

  AdvancedSimilarity2DTransform( const Self & ); //purposely not implemented
  void operator=( const Self & );                //purposely not implemented

  ScaleType m_Scale;

};

//class AdvancedSimilarity2DTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedSimilarity2DTransform.hxx"
#endif

#endif /* __itkAdvancedSimilarity2DTransform_h */
