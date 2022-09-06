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
  Module:    $RCSfile: itkAdvancedEuler3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.14 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedEuler3DTransform_h
#define itkAdvancedEuler3DTransform_h

#include <iostream>
#include "itkAdvancedRigid3DTransform.h"

namespace itk
{

/** \class AdvancedEuler3DTransform
 *
 * \brief AdvancedEuler3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation and translation to the space given 3 euler
 * angles and a 3D translation. Rotation is about a user specified center.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 6 elements.
 * The first 3 represents three euler angle of rotation respectively about
 * the X, Y and Z axis. The last 3 parameters defines the translation in each
 * dimension.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 */
template <class TScalarType = double>
// Data type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedEuler3DTransform : public AdvancedRigid3DTransform<TScalarType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedEuler3DTransform);

  /** Standard class typedefs. */
  using Self = AdvancedEuler3DTransform;
  using Superclass = AdvancedRigid3DTransform<TScalarType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedEuler3DTransform, AdvancedRigid3DTransform);

  /** Dimension of the space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 6);

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

  /** Set/Get the transformation from a container of parameters
   * This is typically used by optimizers.  There are 6 parameters. The first
   * three represent the angles to rotate around the coordinate axis, and the
   * last three represents the offset. */
  void
  SetParameters(const ParametersType & parameters) override;

  const ParametersType &
  GetParameters() const override;

  /** Set the rotational part of the transform. */
  void
  SetRotation(ScalarType angleX, ScalarType angleY, ScalarType angleZ);

  itkGetConstMacro(AngleX, ScalarType);
  itkGetConstMacro(AngleY, ScalarType);
  itkGetConstMacro(AngleZ, ScalarType);

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  /** Set/Get the order of the computation. Default ZXY */
  itkSetMacro(ComputeZYX, bool);
  itkGetConstMacro(ComputeZYX, bool);

  void
  SetIdentity() override;

protected:
  AdvancedEuler3DTransform();
  ~AdvancedEuler3DTransform() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Compute the components of the rotation matrix in the superclass. */
  void
  ComputeMatrix() override;

  void
  ComputeMatrixParameters() override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void
  PrecomputeJacobianOfSpatialJacobian();

private:
  ScalarType m_AngleX;
  ScalarType m_AngleY;
  ScalarType m_AngleZ;
  bool       m_ComputeZYX;
};

// class AdvancedEuler3DTransform

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedEuler3DTransform.hxx"
#endif

#endif /* itkAdvancedEuler3DTransform_h */
