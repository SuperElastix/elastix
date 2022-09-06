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
  Module:    $RCSfile: itkAdvancedSimilarity3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-08-09 04:35:32 $
  Version:   $Revision: 1.3 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedSimilarity3DTransform_h
#define itkAdvancedSimilarity3DTransform_h

#include <iostream>
#include "itkAdvancedVersorRigid3DTransform.h"

namespace itk
{

/** \brief AdvancedSimilarity3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation, translation and isotropic scaling to the space.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 7 elements.
 * The first 3 elements are the components of the versor representation
 * of 3D rotation. The next 3 parameters defines the translation in each
 * dimension. The last parameter defines the isotropic scaling.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 *
 * \sa VersorRigid3DTransform
 */
template <class TScalarType = double>
// Data type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedSimilarity3DTransform : public AdvancedVersorRigid3DTransform<TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedSimilarity3DTransform);

  /** Standard class typedefs. */
  using Self = AdvancedSimilarity3DTransform;
  using Superclass = AdvancedVersorRigid3DTransform<TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedSimilarity3DTransform, AdvancedVersorRigid3DTransform);

  /** Dimension of parameters. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 7);

  /** Parameters Type   */
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::ScalarType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::MatrixType;
  using typename Superclass::InverseMatrixType;
  using typename Superclass::CenterType;
  using typename Superclass::OffsetType;
  using typename Superclass::TranslationType;

  /** Versor type. */
  using typename Superclass::VersorType;
  using typename Superclass::AxisType;
  using typename Superclass::AngleType;
  using ScaleType = TScalarType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Directly set the rotation matrix of the transform.
   * \warning The input matrix must be orthogonal with isotropic scaling
   * to within a specified tolerance, else an exception is thrown.
   *
   * \sa MatrixOffsetTransformBase::SetMatrix() */
  void
  SetMatrix(const MatrixType & matrix) override;

  /** Set the transformation from a container of parameters This is typically
   * used by optimizers.  There are 7 parameters. The first three represent the
   * versor, the next three represent the translation and the last one
   * represents the scaling factor. */
  void
  SetParameters(const ParametersType & parameters) override;

  const ParametersType &
  GetParameters() const override;

  /** Set/Get the value of the isotropic scaling factor */
  void
  SetScale(ScaleType scale);

  itkGetConstReferenceMacro(Scale, ScaleType);

  /** This method computes the Jacobian matrix of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

protected:
  AdvancedSimilarity3DTransform(unsigned int outputSpaceDim, unsigned int paramDim);
  AdvancedSimilarity3DTransform(const MatrixType & matrix, const OutputVectorType & offset);
  AdvancedSimilarity3DTransform();
  ~AdvancedSimilarity3DTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Recomputes the matrix by calling the Superclass::ComputeMatrix() and then
   * applying the scale factor. */
  void
  ComputeMatrix() override;

  /** Computes the parameters from an input matrix. */
  void
  ComputeMatrixParameters() override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void
  PrecomputeJacobianOfSpatialJacobian();

private:
  ScaleType m_Scale;
};

// class AdvancedSimilarity3DTransform

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedSimilarity3DTransform.hxx"
#endif

#endif /* itkAdvancedSimilarity3DTransform_h */
