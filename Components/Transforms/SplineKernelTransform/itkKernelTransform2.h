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
Module:    $RCSfile: itkKernelTransform2.h,v $
Language:  C++
Date:      $Date: 2006-11-28 14:22:18 $
Version:   $Revision: 1.1 $

Copyright (c) Insight Software Consortium. All rights reserved.
See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkKernelTransform2_h
#define itkKernelTransform2_h

#include "itkAdvancedTransform.h"
#include "itkPoint.h"
#include "itkVector.h"
#include "itkMatrix.h"
#include "itkPointSet.h"
#include <deque>
#include <math.h>
#include <vnl/vnl_matrix_fixed.h>
#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_vector_fixed.h>
#include <vnl/vnl_sample.h>
#include <vnl/algo/vnl_svd.h>
#include <vnl/algo/vnl_qr.h>

namespace itk
{

/** \class KernelTransform2
 * Intended to be a base class for elastic body spline and thin plate spline.
 * This is implemented in as straightforward a manner as possible from the
 * IEEE TMI paper by Davis, Khotanzad, Flamig, and Harms, Vol. 16,
 * No. 3 June 1997. Notation closely follows their paper, so if you have it
 * in front of you, this code will make a lot more sense.
 *
 * KernelTransform2:
 *  Provides support for defining source and target landmarks
 *  Defines a number of data types used in the computations
 *  Defines the mathematical framework used to compute all splines,
 *    so that subclasses need only provide a kernel specific to
 *    that spline
 *
 * This formulation allows the stiffness of the spline to
 * be adjusted, allowing the spline to vary from interpolating the
 * landmarks to approximating the landmarks.  This part of the
 * formulation is based on the short paper by R. Sprengel, K. Rohr,
 * H. Stiehl. "Thin-Plate Spline Approximation for Image
 * Registration". In 18th International Conference of the IEEE
 * Engineering in Medicine and Biology Society. 1996.
 *
 * This class was modified to support its use in the ITK registration framework
 * by Rupert Brooks, McGill Centre for Intelligent Machines, Montreal, Canada
 * March 2007.  See the Insight Journal Paper  by Brooks, R., Arbel, T.
 * "Improvements to the itk::KernelTransform and its subclasses."
 *
 * Modified to include it in elastix:
 * - style
 * - make it inherit from AdvancedTransform
 * - make it threadsafe, like was done in the itk as well.
 * - Support for matrix inversion by QR decomposition, instead of SVD.
 *   QR is much faster. Used in SetParameters() and SetFixedParameters().
 * - Much faster Jacobian computation for some of the derived kernel transforms.
 *
 * \ingroup Transforms
 *
 */

template <class TScalarType, // probably only float and double make sense here
          unsigned int NDimensions>
// Number of dimensions
class ITK_TEMPLATE_EXPORT KernelTransform2 : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(KernelTransform2);

  /** Standard class typedefs. */
  using Self = KernelTransform2;
  using Superclass = AdvancedTransform<TScalarType, NDimensions, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(KernelTransform2, AdvancedTransform);

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro(Self);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** Typedefs. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;

  /** AdvancedTransform typedefs. */
  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** PointList typedef. This type is used for maintaining lists of points,
   * specifically, the source and target landmark lists.
   */
  using PointSetTraitsType = DefaultStaticMeshTraits<TScalarType, NDimensions, NDimensions, TScalarType, TScalarType>;
  using PointSetType = PointSet<InputPointType, NDimensions, PointSetTraitsType>;
  using PointSetPointer = typename PointSetType::Pointer;
  using PointsContainer = typename PointSetType::PointsContainer;
  using PointsIterator = typename PointSetType::PointsContainerIterator;
  using PointsConstIterator = typename PointSetType::PointsContainerConstIterator;

  /** VectorSet typedef. */
  using VectorSetType = VectorContainer<unsigned long, InputVectorType>;
  using VectorSetPointer = typename VectorSetType::Pointer;

  /** 'I' (identity) matrix typedef. */
  using IMatrixType = vnl_matrix_fixed<TScalarType, NDimensions, NDimensions>;

  /** Return the number of parameters that completely define the Transform. */
  NumberOfParametersType
  GetNumberOfParameters() const override
  {
    return (this->m_SourceLandmarks->GetNumberOfPoints() * SpaceDimension);
  }


  /** Get the source landmarks list, which we will denote \f$ p \f$. */
  itkGetModifiableObjectMacro(SourceLandmarks, PointSetType);

  /** Set the source landmarks list. */
  virtual void
  SetSourceLandmarks(PointSetType *);

  /** Get the target landmarks list, which we will denote  \f$ q \f$. */
  itkGetModifiableObjectMacro(TargetLandmarks, PointSetType);

  /** Set the target landmarks list. */
  virtual void
  SetTargetLandmarks(PointSetType *);

  /** Get the displacements list, which we will denote \f$ d \f$,
   * where \f$ d_i = q_i - p_i \f$.
   */
  itkGetModifiableObjectMacro(Displacements, VectorSetType);

  /** Compute W matrix. */
  void
  ComputeWMatrix();

  /** Compute L matrix inverse. */
  void
  ComputeLInverse();

  /** Compute the position of point in the new space */
  OutputPointType
  TransformPoint(const InputPointType & thisPoint) const override;

  /** These vector transforms are not implemented for this transform. */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVectorType &) is not implemented for KernelTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVnlVectorType &) is not implemented for KernelTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(
      << "TransformCovariantVector(const InputCovariantVectorType &) is not implemented for KernelTransform");
  }


  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override;

  /** Set the Transformation Parameters to be an identity transform. */
  virtual void
  SetIdentity();

  /** Set the Transformation Parameters and update the internal transformation.
   * The parameters represent the source landmarks. Each landmark point is represented
   * by NDimensions doubles. All the landmarks are concatenated to form one flat
   * Array<double>.
   */
  void
  SetParameters(const ParametersType &) override;

  /** Set Transform Fixed Parameters:
   *     To support the transform file writer this function was
   *     added to set the target landmarks similar to the
   *     SetParameters function setting the source landmarks
   */
  void
  SetFixedParameters(const ParametersType &) override;

  /** Update the Parameters array from the landmarks coordinates. */
  virtual void
  UpdateParameters();

  /** Get the Transformation Parameters - Gets the source landmarks. */
  const ParametersType &
  GetParameters() const override;

  /** Get Transform Fixed Parameters - Gets the target landmarks. */
  const ParametersType &
  GetFixedParameters() const override;

  /** Stiffness of the spline.  A stiffness of zero results in the
   * standard interpolating spline.  A non-zero stiffness allows the
   * spline to approximate rather than interpolate the landmarks.
   * Stiffness values are usually rather small, typically in the range
   * of 0.001 to 0.1. The approximating spline formulation is based on
   * the short paper by R. Sprengel, K. Rohr, H. Stiehl. "Thin-Plate
   * Spline Approximation for Image Registration". In 18th
   * International Conference of the IEEE Engineering in Medicine and
   * Biology Society. 1996.
   */
  virtual void
  SetStiffness(double stiffness)
  {
    this->m_Stiffness = stiffness > 0 ? stiffness : 0.0;
    this->m_LMatrixComputed = false;
    this->m_LInverseComputed = false;
    this->m_WMatrixComputed = false;
  }


  itkGetConstMacro(Stiffness, double);

  /** This method makes only sense for the ElasticBody splines.
   * Declare here, so that you can always call it if you don't know
   * the type of kernel beforehand. It will be overridden in the
   * ElasticBodySplineKernelTransform and in the
   * ElasticBodyReciprocalSplineKernelTransform.
   */
  virtual void
  SetAlpha(TScalarType itkNotUsed(Alpha))
  {}

  /** This method makes only sense for the ElasticBody splines.
   * Declare here, so that you can always call it if you don't know
   * the type of kernel beforehand. It will be overridden in the
   * ElasticBodySplineKernelTransform and in the
   * ElasticBodyReciprocalSplineKernelTransform.
   */
  itkSetMacro(PoissonRatio, TScalarType);
  virtual const TScalarType
  GetPoissonRatio() const
  {
    return this->m_PoissonRatio;
  }


  /** Matrix inversion by SVD or QR decomposition. */
  itkSetMacro(MatrixInversionMethod, std::string);
  itkGetConstReferenceMacro(MatrixInversionMethod, std::string);

  /** Must be provided. */
  void
  GetSpatialJacobian(const InputPointType & inputPoint, SpatialJacobianType & sj) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


  void
  GetSpatialHessian(const InputPointType & inputPoint, SpatialHessianType & sh) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for KernelTransform2");
  }


protected:
  KernelTransform2();
  ~KernelTransform2() override;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

public:
  /** 'G' matrix typedef. */
  using GMatrixType = vnl_matrix_fixed<TScalarType, NDimensions, NDimensions>;

  /** 'L' matrix typedef. */
  using LMatrixType = vnl_matrix<TScalarType>;

  /** 'K' matrix typedef. */
  using KMatrixType = vnl_matrix<TScalarType>;

  /** 'P' matrix typedef. */
  using PMatrixType = vnl_matrix<TScalarType>;

  /** 'Y' matrix typedef. */
  using YMatrixType = vnl_matrix<TScalarType>;

  /** 'W' matrix typedef. */
  using WMatrixType = vnl_matrix<TScalarType>;

  /** 'D' matrix typedef. Deformation component */
  using DMatrixType = vnl_matrix<TScalarType>;

  /** 'A' matrix typedef. Rotational part of the Affine component */
  using AMatrixType = vnl_matrix_fixed<TScalarType, NDimensions, NDimensions>;

  /** 'B' matrix typedef. Translational part of the Affine component */
  using BMatrixType = vnl_vector_fixed<TScalarType, NDimensions>;

  /** Row matrix typedef. */
  using RowMatrixType = vnl_matrix_fixed<TScalarType, 1, NDimensions>;

  /** Column matrix typedef. */
  using ColumnMatrixType = vnl_matrix_fixed<TScalarType, NDimensions, 1>;

  /** The list of source landmarks, denoted 'p'. */
  PointSetPointer m_SourceLandmarks;

  /** The list of target landmarks, denoted 'q'. */
  PointSetPointer m_TargetLandmarks;

protected:
  /** Compute G(x)
   * This is essentially the kernel of the transform.
   * By overriding this method, we can obtain (among others):
   *    Elastic body spline
   *    Thin plate spline
   *    Volume spline.
   */
  virtual void
  ComputeG(const InputVectorType & landmarkVector, GMatrixType & GMatrix) const;

  /** Compute a G(x) for a point to itself (i.e. for the block
   * diagonal elements of the matrix K. Parameter indicates for which
   * landmark the reflexive G is to be computed. The default
   * implementation for the reflexive contribution is a diagonal
   * matrix where the diagonal elements are the stiffness of the
   * spline.
   */
  virtual void
  ComputeReflexiveG(PointsIterator, GMatrixType & GMatrix) const;

  /** Compute the contribution of the landmarks weighted by the kernel
   * function to the global deformation of the space.
   */
  virtual void
  ComputeDeformationContribution(const InputPointType & inputPoint, OutputPointType & result) const;

  /** Compute K matrix. */
  void
  ComputeK();

  /** Compute L matrix. */
  void
  ComputeL();

  /** Compute P matrix. */
  void
  ComputeP();

  /** Compute Y matrix. */
  void
  ComputeY();

  /** Compute displacements \f$ q_i - p_i \f$. */
  void
  ComputeD();

  /** Reorganize the components of W into D (deformable), A (rotation part
   * of affine) and B (translational part of affine ) components.
   * \warning This method release the memory of the W Matrix.
   */
  void
  ReorganizeW();

  /** Stiffness parameter. */
  double m_Stiffness;

  /** The list of displacements.
   * d[i] = q[i] - p[i];
   */
  VectorSetPointer m_Displacements;

  /** The L matrix. */
  LMatrixType m_LMatrix;

  /** The inverse of L, which we also cache. */
  LMatrixType m_LMatrixInverse;

  /** The K matrix. */
  KMatrixType m_KMatrix;

  /** The P matrix. */
  PMatrixType m_PMatrix;

  /** The Y matrix. */
  YMatrixType m_YMatrix;

  /** The W matrix. */
  WMatrixType m_WMatrix;

  /** The Deformation matrix.
   * This is an auxiliary matrix that will hold the Deformation (non-affine)
   * part of the transform. Those are the coefficients that will multiply the
   * Kernel function.
   */
  DMatrixType m_DMatrix;

  /** Rotational/Shearing part of the Affine component of the Transformation. */
  AMatrixType m_AMatrix;

  /** Translational part of the Affine component of the Transformation. */
  BMatrixType m_BVector;

  /** The G matrix.
   * It used to be mutable because m_GMatrix was made an ivar only to avoid
   * copying the matrix at return time but this is not necessary.
   * SK: we don't need this matrix anymore as a member.
   */
  // GMatrixType m_GMatrix;

  /** Has the W matrix been computed? */
  bool m_WMatrixComputed;
  /** Has the L matrix been computed? */
  bool m_LMatrixComputed;
  /** Has the L inverse matrix been computed? */
  bool m_LInverseComputed;
  /** Has the L matrix decomposition been computed? */
  bool m_LMatrixDecompositionComputed;

  /** Decompositions, needed for the L matrix.
   * These decompositions are cached for performance reasons during registration.
   * During registration, in every iteration SetParameters() is called, which in
   * turn calls ComputeWMatrix(). The L matrix is not changed however, and therefore
   * it is not needed to redo the decomposition.
   */
  using SVDDecompositionType = vnl_svd<ScalarType>;
  using QRDecompositionType = vnl_qr<ScalarType>;

  SVDDecompositionType * m_LMatrixDecompositionSVD;
  QRDecompositionType *  m_LMatrixDecompositionQR;

  /** Identity matrix. */
  IMatrixType m_I;

  /** Precomputed nonzero Jacobian indices (simply all params) */
  NonZeroJacobianIndicesType m_NonZeroJacobianIndices;

  /** The Jacobian can be computed much faster for some of the
   * derived kerbel transforms, most notably the TPS.
   */
  bool m_FastComputationPossible;

private:
  TScalarType m_PoissonRatio;

  /** Using SVD or QR decomposition. */
  std::string m_MatrixInversionMethod;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkKernelTransform2.hxx"
#endif

#endif // itkKernelTransform2_h
