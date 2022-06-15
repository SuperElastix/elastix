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

/*

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkAdvancedMatrixOffsetTransformBase.h,v $
  Language:  C++
  Date:      $Date: 2008-06-29 12:58:58 $
  Version:   $Revision: 1.20 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedMatrixOffsetTransformBase_h
#define itkAdvancedMatrixOffsetTransformBase_h

#include <iostream>

#include "itkMatrix.h"
#include "itkAdvancedTransform.h"
#include "itkMacro.h"
#include "itkMacro.h"

namespace itk
{

/**
 * Matrix and Offset transformation of a vector space (e.g. space coordinates)
 *
 * This class serves as a base class for transforms that can be expressed
 * as a linear transformation plus a constant offset (e.g., affine, similarity
 * and rigid transforms).   This base class also provides the concept of
 * using a center of rotation and a translation instead of an offset.
 *
 * As derived instances of this class are specializations of an affine
 * transform, any two of these transformations may be composed and the result
 * is an affine transformation.  However, the order is important.
 * Given two affine transformations T1 and T2, we will say that
 * "precomposing T1 with T2" yields the transformation which applies
 * T1 to the source, and then applies T2 to that result to obtain the
 * target.  Conversely, we will say that "postcomposing T1 with T2"
 * yields the transformation which applies T2 to the source, and then
 * applies T1 to that result to obtain the target.  (Whether T1 or T2
 * comes first lexicographically depends on whether you choose to
 * write mappings from right-to-left or vice versa; we avoid the whole
 * problem by referring to the order of application rather than the
 * textual order.)
 *
 * There are three template parameters for this class:
 *
 * ScalarT       The type to be used for scalar numeric values.  Either
 *               float or double.
 *
 * NInputDimensions   The number of dimensions of the input vector space.
 *
 * NOutputDimensions   The number of dimensions of the output vector space.
 *
 * This class provides several methods for setting the matrix and offset
 * defining the transform. To support the registration framework, the
 * transform parameters can also be set as an Array<double> of size
 * (NInputDimension + 1) * NOutputDimension using method SetParameters().
 * The first (NOutputDimension x NInputDimension) parameters defines the
 * matrix in row-major order (where the column index varies the fastest).
 * The last NOutputDimension parameters defines the translation
 * in each dimensions.
 *
 * \ingroup Transforms
 *
 */

template <class TScalarType = double,        // Data type for scalars
          unsigned int NInputDimensions = 3, // Number of dimensions in the input space
          unsigned int NOutputDimensions = 3>
// Number of dimensions in the output space
class ITK_TEMPLATE_EXPORT AdvancedMatrixOffsetTransformBase
  : public AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  /** Standard typedefs   */
  using Self = AdvancedMatrixOffsetTransformBase;
  using Superclass = AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedMatrixOffsetTransformBase, AdvancedTransform);

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro(ParametersDimension, unsigned int, NOutputDimensions *(NInputDimensions + 1));

  /** Typedefs from the Superclass. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedParametersType;

  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformCategoryEnum;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Standard matrix type for this class. */
  using MatrixType = Matrix<TScalarType, Self::OutputSpaceDimension, Self::InputSpaceDimension>;

  /** Standard inverse matrix type for this class. */
  using InverseMatrixType = Matrix<TScalarType, Self::InputSpaceDimension, Self::OutputSpaceDimension>;

  /** Typedefs. */
  using CenterType = InputPointType;
  using OffsetType = OutputVectorType;
  using TranslationType = OutputVectorType;

  /** Set the transformation to an Identity
   * This sets the matrix to identity and the Offset to null.
   */
  virtual void
  SetIdentity();

  /** Set matrix of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the matrix of an AdvancedMatrixOffsetTransformBase to a
   * value specified by the user.
   *
   * This updates the Offset wrt to current translation
   * and center.  See the warning regarding offset-versus-translation
   * in the documentation for SetCenter.
   *
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  virtual void
  SetMatrix(const MatrixType & matrix)
  {
    this->m_Matrix = matrix;
    this->ComputeOffset();
    this->ComputeMatrixParameters();
    this->m_MatrixMTime.Modified();
    this->Modified();
  }


  /** Get matrix of an AdvancedMatrixOffsetTransformBase
   *
   * This method returns the value of the matrix of the
   * AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const MatrixType &
  GetMatrix() const
  {
    return this->m_Matrix;
  }


  /** Set center of rotation of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the center of rotation of an AdvancedMatrixOffsetTransformBase
   * to a fixed point - for most transforms derived from this class,
   * this point is not a "parameter" of the transform - the exception is that
   * "centered" transforms have center as a parameter during optimization.
   *
   * This method updates offset wrt to current translation and matrix.
   * That is, changing the center changes the transform!
   *
   * WARNING: When using the Center, we strongly recommend only changing the
   * matrix and translation to define a transform.   Changing a transform's
   * center, changes the mapping between spaces - specifically, translation is
   * not changed with respect to that new center, and so the offset is updated
   * to * maintain the consistency with translation.   If a center is not used,
   * or is set before the matrix and the offset, then it is safe to change the
   * offset directly.
   *        As a rule of thumb, if you wish to set the center explicitly, set
   * before Offset computations are done.
   *
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  void
  SetCenter(const InputPointType & center)
  {
    this->m_Center = center;
    this->ComputeOffset();
    this->Modified();
  }


  /** Get center of rotation of the AdvancedMatrixOffsetTransformBase
   *
   * This method returns the point used as the fixed
   * center of rotation for the AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const InputPointType &
  GetCenter() const
  {
    return this->m_Center;
  }


  /** Set translation of an AdvancedMatrixOffsetTransformBase
   *
   * This method sets the translation of an AdvancedMatrixOffsetTransformBase.
   * This updates Offset to reflect current translation.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  void
  SetTranslation(const OutputVectorType & translation)
  {
    this->m_Translation = translation;
    this->ComputeOffset();
    this->Modified();
  }


  /** Get translation component of the AdvancedMatrixOffsetTransformBase
   *
   * This method returns the translation used after rotation
   * about the center point.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const OutputVectorType &
  GetTranslation() const
  {
    return this->m_Translation;
  }


  /** Set the transformation from a container of parameters.
   * The first (NOutputDimension x NInputDimension) parameters define the
   * matrix and the last NOutputDimension parameters the translation.
   * Offset is updated based on current center.
   */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Get the Transformation Parameters. */
  const ParametersType &
  GetParameters() const override;

  /** Set the fixed parameters and update internal transformation. */
  void
  SetFixedParameters(const FixedParametersType &) override;

  /** Get the Fixed Parameters. */
  const FixedParametersType &
  GetFixedParameters() const override;

  /** Transform by an affine transformation
   *
   * This method applies the affine transform given by self to a
   * given point or vector, returning the transformed point or
   * vector.  The TransformPoint method transforms its argument as
   * an affine point, whereas the TransformVector method transforms
   * its argument as a vector.
   */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  OutputVectorType
  TransformVector(const InputVectorType & vector) const override;

  OutputVnlVectorType
  TransformVector(const InputVnlVectorType & vector) const override;

  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType & vector) const override;

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
   * spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

protected:
  /** Construct an AdvancedMatrixOffsetTransformBase object
   *
   * This method constructs a new AdvancedMatrixOffsetTransformBase object and
   * initializes the matrix and offset parts of the transformation
   * to values specified by the caller.  If the arguments are
   * omitted, then the AdvancedMatrixOffsetTransformBase is initialized to an identity
   * transformation in the appropriate number of dimensions.
   */
  explicit AdvancedMatrixOffsetTransformBase(const unsigned int paramDims = ParametersDimension);

  /** Destroy an AdvancedMatrixOffsetTransformBase object. */
  ~AdvancedMatrixOffsetTransformBase() override = default;

  /** Print contents of an AdvancedMatrixOffsetTransformBase. */
  void
  PrintSelf(std::ostream & s, Indent indent) const override;

  virtual void
  ComputeMatrixParameters();

  virtual void
  ComputeMatrix();

  void
  SetVarMatrix(const MatrixType & matrix)
  {
    this->m_Matrix = matrix;
    this->m_MatrixMTime.Modified();
  }

  void
  ComputeTranslation();

  void
  SetVarTranslation(const OutputVectorType & translation)
  {
    this->m_Translation = translation;
  }

  virtual void
  ComputeOffset();

  /** Get offset of an AdvancedMatrixOffsetTransformBase
   *
   * This method returns the offset value of the AdvancedMatrixOffsetTransformBase.
   * To define an affine transform, you must set the matrix,
   * center, and translation OR the matrix and offset.
   */
  const OutputVectorType &
  GetOffset() const
  {
    return this->m_Offset;
  }

  /** (spatial) Jacobians and Hessians can mostly be precomputed by this transform.
   * Store them in these member variables.
   * SpatialJacobian is simply m_Matrix */
  NonZeroJacobianIndicesType    m_NonZeroJacobianIndices;
  SpatialHessianType            m_SpatialHessian;
  JacobianOfSpatialJacobianType m_JacobianOfSpatialJacobian;
  JacobianOfSpatialHessianType  m_JacobianOfSpatialHessian;

private:
  AdvancedMatrixOffsetTransformBase(const Self & other);
  const Self &
  operator=(const Self &);

  /** Called by constructors: */
  void
  PrecomputeJacobians(unsigned int paramDims);

  const InverseMatrixType &
  GetInverseMatrix() const;


  /** Member variables. */
  MatrixType                m_Matrix{ MatrixType::GetIdentity() };               // Matrix of the transformation
  OutputVectorType          m_Offset{};                                          // Offset of the transformation
  mutable InverseMatrixType m_InverseMatrix{ InverseMatrixType::GetIdentity() }; // Inverse of the matrix
  mutable bool              m_Singular{ false };                                 // Is m_Inverse singular?

  InputPointType   m_Center{};
  OutputVectorType m_Translation{};

  /** To avoid recomputation of the inverse if not needed. */
  TimeStamp         m_MatrixMTime;
  mutable TimeStamp m_InverseMatrixMTime;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedMatrixOffsetTransformBase.hxx"
#endif

#endif /* itkAdvancedMatrixOffsetTransformBase_h */
