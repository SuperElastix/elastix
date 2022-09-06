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
  Module:    $RCSfile: itkAdvancedBSplineDeformableTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-04-11 16:28:11 $
  Version:   $Revision: 1.38 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkAdvancedBSplineDeformableTransform_h
#define itkAdvancedBSplineDeformableTransform_h

#include "itkAdvancedBSplineDeformableTransformBase.h"

#include "itkImage.h"
#include "itkImageRegion.h"
#include "itkBSplineInterpolationWeightFunction2.h"
#include "itkBSplineInterpolationDerivativeWeightFunction.h"
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"

namespace itk
{

// Forward declarations for friendship
template <class TScalarType, unsigned int NDimensions, unsigned int VSplineOrder>
class ITK_TEMPLATE_EXPORT MultiBSplineDeformableTransformWithNormal;

/** \class AdvancedBSplineDeformableTransform
 * \brief Deformable transform using a B-spline representation
 *
 * This class encapsulates a deformable transform of points from one
 * N-dimensional one space to another N-dimensional space.
 * The deformation field is modeled using B-splines.
 * A deformation is defined on a sparse regular grid of control points
 * \f$ \vec{\lambda}_j \f$ and is varied by defining a deformation
 * \f$ \vec{g}(\vec{\lambda}_j) \f$ of each control point.
 * The deformation \f$ D(\vec{x}) \f$ at any point \f$ \vec{x} \f$
 * is obtained by using a B-spline interpolation kernel.
 *
 * The deformation field grid is defined by a user specified GridRegion,
 * GridSpacing and GridOrigin. Each grid/control point has associated with it
 * N deformation coefficients \f$ \vec{\delta}_j \f$, representing the N
 * directional components of the deformation. Deformation outside the grid
 * plus support region for the B-spline interpolation is assumed to be zero.
 *
 * Additionally, the user can specified an addition bulk transform \f$ B \f$
 * such that the transformed point is given by:
 * \f[ \vec{y} = B(\vec{x}) + D(\vec{x}) \f]
 *
 * The parameters for this transform is N x N-D grid of spline coefficients.
 * The user specifies the parameters as one flat array: each N-D grid
 * is represented by an array in the same way an N-D image is represented
 * in the buffer; the N arrays are then concatentated together on form
 * a single array.
 *
 * For efficiency, this transform does not make a copy of the parameters.
 * It only keeps a pointer to the input parameters and assumes that the memory
 * is managed by the caller.
 *
 * The following illustrates the typical usage of this class:
 * \verbatim
 * typedef AdvancedBSplineDeformableTransform<double,2,3> TransformType;
 * auto transform = TransformType::New();
 *
 * transform->SetGridRegion( region );
 * transform->SetGridSpacing( spacing );
 * transform->SetGridOrigin( origin );
 *
 * // NB: the region must be set first before setting the parameters
 *
 * TransformType::ParametersType parameters(
 *                                       transform->GetNumberOfParameters() );
 *
 * // Fill the parameters with values
 *
 * transform->SetParameters( parameters )
 *
 * outputPoint = transform->TransformPoint( inputPoint );
 *
 * \endverbatim
 *
 * An alternative way to set the B-spline coefficients is via array of
 * images. The grid region, spacing and origin information is taken
 * directly from the first image. It is assumed that the subsequent images
 * are the same buffered region. The following illustrates the API:
 * \verbatim
 *
 * TransformType::ImageConstPointer images[2];
 *
 * // Fill the images up with values
 *
 * transform->SetCoefficientImages( images );
 * outputPoint = transform->TransformPoint( inputPoint );
 *
 * \endverbatim
 *
 * Warning: use either the SetParameters() or SetCoefficientImages()
 * API. Mixing the two modes may results in unexpected results.
 *
 * The class is templated coordinate representation type (float or double),
 * the space dimension and the spline order.
 *
 * \ingroup Transforms
 */
template <class TScalarType = double,   // Data type for scalars
          unsigned int NDimensions = 3, // Number of dimensions
          unsigned int VSplineOrder = 3>
// Spline order
class ITK_TEMPLATE_EXPORT AdvancedBSplineDeformableTransform
  : public AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedBSplineDeformableTransform);

  /** Standard class typedefs. */
  using Self = AdvancedBSplineDeformableTransform;
  using Superclass = AdvancedBSplineDeformableTransformBase<TScalarType, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedBSplineDeformableTransform, AdvancedBSplineDeformableTransformBase);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** The B-spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Typedefs from Superclass. */
  using typename Superclass::ParametersType;
  using typename Superclass::ParametersValueType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::DerivativeType;
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

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MovingImageGradientValueType;

  /** Parameters as SpaceDimension number of images. */
  using typename Superclass::PixelType;
  using typename Superclass::ImageType;
  using typename Superclass::ImagePointer;

  /** Typedefs for specifying the extend to the grid. */
  using typename Superclass::RegionType;

  using typename Superclass::IndexType;
  using typename Superclass::SizeType;
  using typename Superclass::SpacingType;
  using typename Superclass::DirectionType;
  using typename Superclass::OriginType;
  using typename Superclass::GridOffsetType;

  /** This method specifies the region over which the grid resides. */
  void
  SetGridRegion(const RegionType & region) override;

  /** Transform points by a B-spline deformable transformation. */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /** Interpolation weights function type. */
  using WeightsFunctionType = BSplineInterpolationWeightFunction2<ScalarType, Self::SpaceDimension, VSplineOrder>;
  using WeightsFunctionPointer = typename WeightsFunctionType::Pointer;
  using WeightsType = typename WeightsFunctionType::WeightsType;
  using ContinuousIndexType = typename WeightsFunctionType::ContinuousIndexType;
  using DerivativeWeightsFunctionType =
    BSplineInterpolationDerivativeWeightFunction<ScalarType, Self::SpaceDimension, VSplineOrder>;
  using DerivativeWeightsFunctionPointer = typename DerivativeWeightsFunctionType::Pointer;
  using SODerivativeWeightsFunctionType =
    BSplineInterpolationSecondOrderDerivativeWeightFunction<ScalarType, Self::SpaceDimension, VSplineOrder>;
  using SODerivativeWeightsFunctionPointer = typename SODerivativeWeightsFunctionType::Pointer;

  /** Parameter index array type. */
  using typename Superclass::ParameterIndexArrayType;


  /** Get number of weights. */
  unsigned long
  GetNumberOfWeights() const
  {
    return this->m_WeightsFunction->GetNumberOfWeights();
  }


  unsigned int
  GetNumberOfAffectedWeights() const override;

  NumberOfParametersType
  GetNumberOfNonZeroJacobianIndices() const override;

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType & inputPoint, JacobianType & j, NonZeroJacobianIndicesType & nzji) const override;

  /** Compute the inner product of the Jacobian with the moving image gradient.
   * The Jacobian is (partially) constructed inside this function, but not returned.
   */
  void
  EvaluateJacobianWithImageGradientProduct(const InputPointType &          inputPoint,
                                           const MovingImageGradientType & movingImageGradient,
                                           DerivativeType &                imageJacobian,
                                           NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType & inputPoint, SpatialJacobianType & sj) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType & inputPoint, SpatialHessianType & sh) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

protected:
  /** Print contents of an AdvancedBSplineDeformableTransform. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  AdvancedBSplineDeformableTransform();
  ~AdvancedBSplineDeformableTransform() override = default;

  /** Allow subclasses to access and manipulate the weights function. */
  // Why??
  itkSetObjectMacro(WeightsFunction, WeightsFunctionType);
  itkGetModifiableObjectMacro(WeightsFunction, WeightsFunctionType);

  /** Wrap flat array into images of coefficients. */
  void
  WrapAsImages();

  void
  ComputeNonZeroJacobianIndices(NonZeroJacobianIndicesType & nonZeroJacobianIndices,
                                const RegionType &           supportRegion) const override;

  using typename Superclass::JacobianImageType;
  using typename Superclass::JacobianPixelType;

  /** Pointer to function used to compute B-spline interpolation weights.
   * For each direction we create a different weights function for thread-
   * safety.
   */
  WeightsFunctionPointer                                       m_WeightsFunction;
  std::vector<DerivativeWeightsFunctionPointer>                m_DerivativeWeightsFunctions;
  std::vector<std::vector<SODerivativeWeightsFunctionPointer>> m_SODerivativeWeightsFunctions;

private:
  friend class MultiBSplineDeformableTransformWithNormal<ScalarType, Self::SpaceDimension, VSplineOrder>;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedBSplineDeformableTransform.hxx"
#endif

#endif /* itkAdvancedBSplineDeformableTransform_h */
