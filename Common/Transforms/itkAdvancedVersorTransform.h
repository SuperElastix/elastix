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
  Module:    $RCSfile: itkAdvancedVersorTransform.h,v $
  Language:  C++
  Date:      $Date: 2006-08-09 04:35:32 $
  Version:   $Revision: 1.17 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedVersorTransform_h
#define itkAdvancedVersorTransform_h

#include <iostream>
#include "itkAdvancedRigid3DTransform.h"
#include <vnl/vnl_quaternion.h>
#include "itkVersor.h"

namespace itk
{

/**
 *
 * AdvancedVersorTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation to the space. Rotation is about
 * a user specified center.
 *
 * The serialization of the optimizable parameters is an array of 3 elements
 * representing the right part of the versor.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \todo Need to make sure that the translation parameters in the baseclass
 * cannot be set to non-zero values.
 *
 * NB: SK: this class is just to have the AdvancedSimilarity3DTransform. It is not complete.
 *
 * \ingroup Transforms
 *
 **/
template <class TScalarType = double>
// Data type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedVersorTransform : public AdvancedRigid3DTransform<TScalarType>
{
public:
  /** Standard Self Typedef */
  using Self = AdvancedVersorTransform;
  using Superclass = AdvancedRigid3DTransform<TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods).  */
  itkTypeMacro(AdvancedVersorTransform, AdvancedRigid3DTransform);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Dimension of parameters */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 3);

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

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** VnlQuaternion Type */
  using VnlQuaternionType = vnl_quaternion<TScalarType>;

  /** Versor Type */
  using VersorType = Versor<TScalarType>;
  using AxisType = typename VersorType::VectorType;
  using AngleType = typename VersorType::ValueType;

  /**
   * Set the transformation from a container of parameters
   * This is typically used by optimizers.
   *
   * There are 3 parameters. They represent the components
   * of the right part of the versor. This can be seen
   * as the components of the vector parallel to the rotation
   * axis and multiplied by std::sin( angle / 2 ). */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Get the Transformation Parameters. */
  const ParametersType &
  GetParameters() const override;

  /** Set the rotational part of the transform */
  void
  SetRotation(const VersorType & versor);

  void
  SetRotation(const AxisType & axis, AngleType angle);

  itkGetConstReferenceMacro(Versor, VersorType);

  /** Set the parameters to the IdentityTransform */
  void
  SetIdentity() override;

  /** This method computes the Jacobian matrix of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

protected:
  /** Construct an AdvancedVersorTransform object */
  AdvancedVersorTransform(const MatrixType & matrix, const OutputVectorType & offset);
  AdvancedVersorTransform(unsigned int paramDims);
  AdvancedVersorTransform();

  /** Destroy an AdvancedVersorTransform object */
  ~AdvancedVersorTransform() override = default;

  void
  SetVarVersor(const VersorType & newVersor)
  {
    m_Versor = newVersor;
  }

  /** Print contents of a AdvancedVersorTransform */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute Matrix
   *  Compute the components of the rotation matrix in the superclass */
  void
  ComputeMatrix() override;

  void
  ComputeMatrixParameters() override;

private:
  /** Copy a AdvancedVersorTransform object */
  AdvancedVersorTransform(const Self & other); // Not implemented

  /** Assignment operator */
  const Self &
  operator=(const Self &); // Not implemented

  /** Versor containing the rotation */
  VersorType m_Versor;
};

// class AdvancedVersorTransform

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedVersorTransform.hxx"
#endif

#endif /* itkAdvancedVersorTransform_h */
