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
#ifndef itkAffineLogTransform_h
#define itkAffineLogTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/** \class AffineLogTransform
 *
 *
 * \ingroup Transforms
 */
template <class TScalarType = double, unsigned int Dimension = 2> // Data type for scalars (float or double)
class AffineLogTransform : public AdvancedMatrixOffsetTransformBase<TScalarType, Dimension, Dimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineLogTransform);

  /** Standard class typedefs. */
  using Self = AffineLogTransform;
  using Superclass = AdvancedMatrixOffsetTransformBase<TScalarType, Dimension, Dimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AffineLogTransform, AdvancedMatrixOffsetTransformBase);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, Dimension);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, Dimension);
  itkStaticConstMacro(ParametersDimension, unsigned int, (Dimension + 1) * Dimension);

  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::ScalarType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::MatrixType;
  using typename Superclass::InverseMatrixType;
  using typename Superclass::CenterType;
  using typename Superclass::TranslationType;
  using typename Superclass::OffsetType;
  using AngleType = typename Superclass::ScalarType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  using ScalarArrayType = FixedArray<ScalarType>;

  void
  SetParameters(const ParametersType & parameters) override;

  const ParametersType &
  GetParameters() const override;

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  void
  SetIdentity() override;

protected:
  AffineLogTransform();
  AffineLogTransform(const MatrixType & matrix, const OutputPointType & offset);
  AffineLogTransform(unsigned int outputSpaceDims, unsigned int paramsSpaceDims);

  ~AffineLogTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void
  PrecomputeJacobianOfSpatialJacobian();

private:
  MatrixType m_MatrixLogDomain;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAffineLogTransform.hxx"
#endif

#endif /* itkAffineLogTransform_h */
