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
#ifndef itkAdvancedSimilarity2DTransform_h
#define itkAdvancedSimilarity2DTransform_h

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
template <class TScalarType = double>
// Data type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedSimilarity2DTransform : public AdvancedRigid2DTransform<TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedSimilarity2DTransform);

  /** Standard class typedefs. */
  using Self = AdvancedSimilarity2DTransform;
  using Superclass = AdvancedRigid2DTransform<TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedSimilarity2DTransform, AdvancedRigid2DTransform);

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 2);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(ParametersDimension, unsigned int, 4);

  /** Scalar type. */
  using typename Superclass::ScalarType;
  using ScaleType = TScalarType;

  /** Parameters type. */
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;

  /** Jacobian type. */
  using typename Superclass::JacobianType;

  /** Offset type. */
  using typename Superclass::OffsetType;

  /** Matrix type. */
  using typename Superclass::MatrixType;

  /** Point type. */
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;

  /** Vector type. */
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;

  /** CovariantVector type. */
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;

  /** VnlVector type. */
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Set the Scale part of the transform. */
  void
  SetScale(ScaleType scale);

  itkGetConstReferenceMacro(Scale, ScaleType);

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.
   * There are 4 parameters. The first one represents the
   * scale, the second represents the angle of rotation
   * and the last two represent the translation.
   * The center of rotation is fixed.
   *
   * \sa Transform::SetParameters()
   * \sa Transform::SetFixedParameters() */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Get the parameters that uniquely define the transform
   * This is typically used by optimizers.
   * There are 4 parameters. The first one represents the
   * scale, the second represents the angle of rotation,
   * and the last two represent the translation.
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

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  /** Set the transformation to an identity. */
  void
  SetIdentity() override;

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
  void
  SetMatrix(const MatrixType & matrix) override;

protected:
  AdvancedSimilarity2DTransform();
  ~AdvancedSimilarity2DTransform() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute matrix from angle and scale. This is used in Set methods
   * to update the underlying matrix whenever a transform parameter
   * is changed. */
  void
  ComputeMatrix() override;

  /** Compute the angle and scale from the matrix. This is used to compute
   * transform parameters from a given matrix. This is used in
   * MatrixOffsetTransformBase::Compose() and
   * MatrixOffsetTransformBase::GetInverse(). */
  void
  ComputeMatrixParameters() override;

  /** Set the scale without updating underlying variables. */
  void
  SetVarScale(ScaleType scale)
  {
    m_Scale = scale;
  }

  /** Update the m_JacobianOfSpatialJacobian.  */
  void
  PrecomputeJacobianOfSpatialJacobian() override;

private:
  ScaleType m_Scale;
};

// class AdvancedSimilarity2DTransform

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedSimilarity2DTransform.hxx"
#endif

#endif /* itkAdvancedSimilarity2DTransform_h */
