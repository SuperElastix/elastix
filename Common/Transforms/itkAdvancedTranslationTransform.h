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
  Module:    $RCSfile: itkAdvancedTranslationTransform.h,v $
  Language:  C++
  Date:      $Date: 2007-07-15 16:38:25 $
  Version:   $Revision: 1.36 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedTranslationTransform_h
#define itkAdvancedTranslationTransform_h

#include <iostream>
#include "itkAdvancedTransform.h"
#include "itkMacro.h"
#include "itkMatrix.h"

namespace itk
{

/** \brief Translation transformation of a vector space (e.g. space coordinates)
 *
 * The same functionality could be obtained by using the Affine tranform,
 * but with a large difference in performace.
 *
 * \ingroup Transforms
 */
template <class TScalarType = double, // Data type for scalars (float or double)
          unsigned int NDimensions = 3>
// Number of dimensions
class ITK_TEMPLATE_EXPORT AdvancedTranslationTransform : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedTranslationTransform);

  /** Standard class typedefs. */
  using Self = AdvancedTranslationTransform;
  using Superclass = AdvancedTransform<TScalarType, NDimensions, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory.*/
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedTranslationTransform, AdvancedTransform);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, NDimensions);

  /** Standard scalar type for this class. */
  using typename Superclass::ScalarType;

  /** Standard parameters container. */
  using typename Superclass::ParametersType;
  using typename Superclass::FixedParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::TransformCategoryEnum;

  /** Standard Jacobian container. */
  using typename Superclass::JacobianType;

  /** Standard vector type for this class. */
  using InputVectorType = Vector<TScalarType, Self::SpaceDimension>;
  using OutputVectorType = Vector<TScalarType, Self::SpaceDimension>;

  /** Standard covariant vector type for this class. */
  using InputCovariantVectorType = CovariantVector<TScalarType, Self::SpaceDimension>;
  using OutputCovariantVectorType = CovariantVector<TScalarType, Self::SpaceDimension>;

  /** Standard vnl_vector type for this class. */
  using InputVnlVectorType = vnl_vector_fixed<TScalarType, Self::SpaceDimension>;
  using OutputVnlVectorType = vnl_vector_fixed<TScalarType, Self::SpaceDimension>;

  /** Standard coordinate point type for this class. */
  using InputPointType = Point<TScalarType, Self::SpaceDimension>;
  using OutputPointType = Point<TScalarType, Self::SpaceDimension>;

  /** AdvancedTransform typedefs */
  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** This method returns the value of the offset of the
   * AdvancedTranslationTransform.
   */
  const OutputVectorType &
  GetOffset() const
  {
    return m_Offset;
  }

  /** This method sets the parameters for the transform
   * value specified by the user. */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Get the Transformation Parameters. */
  const ParametersType &
  GetParameters() const override;

  /** Set offset of an Translation Transform.
   * This method sets the offset of an AdvancedTranslationTransform to a
   * value specified by the user. */
  void
  SetOffset(const OutputVectorType & offset)
  {
    m_Offset = offset;
    return;
  }

  /** Transform by an affine transformation.
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector. */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  OutputVectorType
  TransformVector(const InputVectorType & vector) const override;

  OutputVnlVectorType
  TransformVector(const InputVnlVectorType & vector) const override;

  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType & vector) const override;

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType &, SpatialJacobianType &) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType &, SpatialHessianType &) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &,
                               JacobianOfSpatialJacobianType &,
                               NonZeroJacobianIndicesType &) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &,
                               SpatialJacobianType &,
                               JacobianOfSpatialJacobianType &,
                               NonZeroJacobianIndicesType &) const override;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &,
                              JacobianOfSpatialHessianType &,
                              NonZeroJacobianIndicesType &) const override;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Set the parameters to the IdentityTransform */
  void
  SetIdentity();

  /** Return the number of parameters that completely define the Transform  */
  NumberOfParametersType
  GetNumberOfParameters() const override
  {
    return NDimensions;
  }

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  bool
  IsLinear() const override
  {
    return true;
  }

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  TransformCategoryEnum
  GetTransformCategory() const override
  {
    return TransformCategoryEnum::Linear;
  }


  /** Set the fixed parameters and update internal transformation.
   * The Translation Transform does not require fixed parameters,
   * therefore the implementation of this method is a null operation. */
  void
  SetFixedParameters(const FixedParametersType &) override
  { /* purposely blank */
  }

  /** Get the Fixed Parameters. The AdvancedTranslationTransform does not
   * require Fixed parameters, therefore this method returns an
   * parameters array of size zero. */
  const FixedParametersType &
  GetFixedParameters() const override
  {
    this->m_FixedParameters.SetSize(0);
    return this->m_FixedParameters;
  }


protected:
  AdvancedTranslationTransform();
  ~AdvancedTranslationTransform() override;
  /** Print contents of an AdvancedTranslationTransform. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  OutputVectorType m_Offset{}; // Offset of the transformation

  JacobianType                        m_LocalJacobian{ JacobianType(NDimensions, NDimensions) };
  const SpatialJacobianType           m_SpatialJacobian{ SpatialJacobianType::GetIdentity() };
  const SpatialHessianType            m_SpatialHessian{};
  NonZeroJacobianIndicesType          m_NonZeroJacobianIndices{ NonZeroJacobianIndicesType(NDimensions) };
  const JacobianOfSpatialJacobianType m_JacobianOfSpatialJacobian{ JacobianOfSpatialJacobianType(NDimensions) };
  const JacobianOfSpatialHessianType  m_JacobianOfSpatialHessian{ JacobianOfSpatialHessianType(NDimensions) };
};

// class AdvancedTranslationTransform


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedTranslationTransform.hxx"
#endif

#endif /* itkAdvancedTranslationTransform_h */
