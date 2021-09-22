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
  Module:    $RCSfile: itkAdvancedRigid3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2007-02-13 21:46:04 $
  Version:   $Revision: 1.38 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedRigid3DTransform_h
#define itkAdvancedRigid3DTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"
#include "itkMacro.h"
#include "itkMatrix.h"
#include "itkVersor.h"

namespace itk
{

/** \brief AdvancedRigid3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation and translation in 3D space.
 * The transform is specified as a rotation matrix around a arbitrary center
 * and is followed by a translation.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 12 elements.
 * The first 9 parameters represents the rotation matrix in column-major order
 * (where the column index varies the fastest). The last 3 parameters defines
 * the translation in each dimension.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation in each dimension.
 *
 * \ingroup Transforms
 */
template <class TScalarType = double>
// type for scalars (float or double)
class ITK_TEMPLATE_EXPORT AdvancedRigid3DTransform : public AdvancedMatrixOffsetTransformBase<TScalarType, 3, 3>
{
public:
  /** Standard class typedefs. */
  typedef AdvancedRigid3DTransform                             Self;
  typedef AdvancedMatrixOffsetTransformBase<TScalarType, 3, 3> Superclass;
  typedef SmartPointer<Self>                                   Pointer;
  typedef SmartPointer<const Self>                             ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedRigid3DTransform, AdvancedMatrixOffsetTransformBase);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** Dimension of the space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);
  itkStaticConstMacro(InputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, 3);
  itkStaticConstMacro(ParametersDimension, unsigned int, 12);

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

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Set the transformation from a container of parameters
   * This is typically used by optimizers.
   * There are 12 parameters. The first 9 represents the rotation
   * matrix is column-major order and the last 3 represents the translation.
   *
   * \warning The rotation matrix must be orthogonal to within a specified tolerance,
   * else an exception is thrown.
   *
   * \sa Transform::SetParameters()
   * \sa Transform::SetFixedParameters() */
  void
  SetParameters(const ParametersType & parameters) override;

  /** Directly set the rotation matrix of the transform.
   * \warning The input matrix must be orthogonal to within a specified tolerance,
   * else an exception is thrown.
   *
   * \sa AdvancedMatrixOffsetTransformBase::SetMatrix() */
  void
  SetMatrix(const MatrixType & matrix) override;

  /**
   * Get rotation Matrix from an AdvancedRigid3DTransform
   *
   * This method returns the value of the rotation of the
   * AdvancedRigid3DTransform.
   *
   * \deprecated Use GetMatrix instead
   **/
  const MatrixType &
  GetRotationMatrix()
  {
    return this->GetMatrix();
  }

  /**
   * Set the rotation Matrix of a Rigid3D Transform
   *
   * This method sets the 3x3 matrix representing a rotation
   * in the transform.  The Matrix is expected to be orthogonal
   * with a certain tolerance.
   *
   * \deprecated Use SetMatrix instead
   *
   **/
  virtual void
  SetRotationMatrix(const MatrixType & matrix)
  {
    this->SetMatrix(matrix);
  }

  /**
   * Compose the transformation with a translation
   *
   * This method modifies self to include a translation of the
   * origin.  The translation is precomposed with self if pre is
   * true, and postcomposed otherwise.
   **/
  void
  Translate(const OffsetType & offset, bool pre = false);

  /**
   * Back transform by an affine transformation
   *
   * This method finds the point or vector that maps to a given
   * point or vector under the affine transformation defined by
   * self.  If no such point exists, an exception is thrown.
   *
   * \deprecated Please use GetInverseTransform and then call the forward
   *   transform using the result.
   *
   **/
  InputPointType
  BackTransform(const OutputPointType & point) const;

  InputVectorType
  BackTransform(const OutputVectorType & vector) const;

  InputVnlVectorType
  BackTransform(const OutputVnlVectorType & vector) const;

  InputCovariantVectorType
  BackTransform(const OutputCovariantVectorType & vector) const;

  /**
   * Utility function to test if a matrix is orthogonal within a specified
   * tolerance
   */
  bool
  MatrixIsOrthogonal(const MatrixType & matrix, double tol = 1e-10);

protected:
  explicit AdvancedRigid3DTransform(unsigned int paramDim);
  AdvancedRigid3DTransform(const MatrixType & matrix, const OutputVectorType & offset);
  AdvancedRigid3DTransform();
  ~AdvancedRigid3DTransform() override;

  /**
   * Print contents of an AdvancedRigid3DTransform
   **/
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  AdvancedRigid3DTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedRigid3DTransform.hxx"
#endif

#endif /* itkAdvancedRigid3DTransform_h */
