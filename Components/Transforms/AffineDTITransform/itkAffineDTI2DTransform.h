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
#ifndef itkAffineDTI2DTransform_h
#define itkAffineDTI2DTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/** \class AffineDTI3DTransform
 *
 * \brief AffineDTI3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies an affine transformation, but is parameterized by
 * angles, shear factors, scales, and translation, instead of by the affine matrix.
 * It is meant for registration of MR diffusion weighted images, but could be
 * used for other images as well of course.
 *
 * The affine model is adopted from the following paper:
 * [1] A. Leemans and D.K. Jones. "The B-matrix must be rotated when correcting for subject motion in DTI data".
 *   Magnetic Resonance in Medicine, Volume 61, Issue 6, pages 1336 - 1349, 2009.
 *
 * The model is as follows:\n
 *   T(x) = R G S (x-c) + t + c\n
 * with:
 *   - R = Rx Ry Rz (rotation matrices)
 *   - G = Gx Gy Gz (shear matrices)
 *   - S = diag( [sx sy sz] ) (scaling matrix)
 *   - c = center of rotation
 *   - t = translation
 * See [1] for exact expressions for Rx, Gx etc.
 *
 * Using this model, the rotation components can be easily extracted an applied
 * to the B-matrix.
 *
 * The parameters are ordered as follows:
 * <tt>in 2D: [ Angle ShearX ShearY ScaleX ScaleY TranslationX TranslationY ]</tt>
 * <tt>in 3D: [ AngleX AngleY AngleZ ShearX ShearY ShearZ ScaleX ScaleY ScaleZ TranslationX TranslationY TranslationZ
 * ]</tt>
 *
 * The serialization of the fixed parameters is an array of 2 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 */
template <class TScalarType = double>
// Data type for scalars (float or double)
class AffineDTI2DTransform : public AdvancedMatrixOffsetTransformBase<TScalarType, 2, 2>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AffineDTI2DTransform);

  /** Standard class typedefs. */
  using Self = AffineDTI2DTransform;
  using Superclass = AdvancedMatrixOffsetTransformBase<TScalarType, 2, 2>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AffineDTI2DTransform, AdvancedMatrixOffsetTransformBase);

  /** Dimension of the space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 2);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 2);
  itkStaticConstMacro(ParametersDimension, unsigned int, 7);

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

  /** Set/Get the transformation from a container of parameters
   * This is typically used by optimizers.  There are 7 parameters.
   * [ R Gx Gy Sx Sy Tx Ty ]
   * ~rotation, scale, skew, translation
   */
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
  AffineDTI2DTransform();
  AffineDTI2DTransform(const MatrixType & matrix, const OutputPointType & offset);
  AffineDTI2DTransform(unsigned int outputSpaceDims, unsigned int paramsSpaceDims);

  ~AffineDTI2DTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Set values of angles etc directly without recomputing other parameters. */
  void
  SetVarAngleScaleShear(ScalarArrayType angle, ScalarArrayType shear, ScalarArrayType scale);

  /** Compute the components of the rotation matrix in the superclass. */
  void
  ComputeMatrix() override;

  void
  ComputeMatrixParameters() override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void
  PrecomputeJacobianOfSpatialJacobian();

private:
  ScalarArrayType m_Angle;
  ScalarArrayType m_Shear;
  ScalarArrayType m_Scale;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAffineDTI2DTransform.hxx"
#endif

#endif /* itkAffineDTI2DTransform_h */
