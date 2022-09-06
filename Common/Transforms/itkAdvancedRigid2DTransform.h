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
#ifndef itkAdvancedRigid2DTransform_h
#define itkAdvancedRigid2DTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkMacro.h"

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
template <class TScalarType = double>
// Data type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedRigid2DTransform
  : public AdvancedMatrixOffsetTransformBase<TScalarType, 2, 2> // Dimensions of input and output spaces
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedRigid2DTransform);

  /** Standard class typedefs. */
  using Self = AdvancedRigid2DTransform;
  using Superclass = AdvancedMatrixOffsetTransformBase<TScalarType, 2, 2>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedRigid2DTransform, AdvancedMatrixOffsetTransformBase);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Dimension of the space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(ParametersDimension, unsigned int, 3);

  /** Scalar type. */
  using typename Superclass::ScalarType;

  /** Parameters type. */
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;

  /** Jacobian type. */
  using typename Superclass::JacobianType;

  /// Standard matrix type for this class
  using typename Superclass::MatrixType;

  /// Standard vector type for this class
  using typename Superclass::OffsetType;

  /// Standard vector type for this class
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;

  /// Standard covariant vector type for this class
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;

  /// Standard vnl_vector type for this class
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;

  /// Standard coordinate point type for this class
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

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
  void
  SetMatrix(const MatrixType & matrix) override;

  /** Set/Get the angle of rotation in radians */
  void
  SetAngle(TScalarType angle);

  itkGetConstReferenceMacro(Angle, TScalarType);

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.
   * There are 3 parameters. The first one represents the
   * angle of rotation in radians and the last two represents the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::SetParameters()
   * \sa Transform::SetFixedParameters() */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Get the parameters that uniquely define the transform
   * This is typically used by optimizers.
   * There are 3 parameters. The first one represents the
   * angle or rotation in radians and the last two represents the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::GetParameters()
   * \sa Transform::GetFixedParameters() */
  const ParametersType &
  GetParameters() const override;

  /** This method computes the Jacobian matrix of the transformation
   * at a given input point.
   *
   * \sa Transform::GetJacobian() */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  /** Reset the parameters to create and identity transform. */
  void
  SetIdentity() override;

protected:
  AdvancedRigid2DTransform();
  AdvancedRigid2DTransform(unsigned int outputSpaceDimension, unsigned int parametersDimension);
  ~AdvancedRigid2DTransform() override = default;

  /**
   * Print contents of an AdvancedRigid2DTransform
   */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute the matrix from angle. This is used in Set methods
   * to update the underlying matrix whenever a transform parameter
   * is changed.
   * Also update the m_JacobianOfSpatialJacobian. */
  void
  ComputeMatrix() override;

  /** Compute the angle from the matrix. This is used to compute
   * transform parameters from a given matrix. This is used in
   * AdvancedMatrixOffsetTransformBase::Compose() and
   * AdvancedMatrixOffsetTransformBase::GetInverse(). */
  void
  ComputeMatrixParameters() override;

  /** Update angle without recomputation of other internal variables. */
  void
  SetVarAngle(TScalarType angle)
  {
    m_Angle = angle;
  }

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void
  PrecomputeJacobianOfSpatialJacobian();

private:
  TScalarType m_Angle;
};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedRigid2DTransform.hxx"
#endif

#endif /* itkAdvancedRigid2DTransform_h */
