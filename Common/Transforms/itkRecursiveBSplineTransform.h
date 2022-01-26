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
#ifndef itkRecursiveBSplineTransform_h
#define itkRecursiveBSplineTransform_h

#include "itkAdvancedBSplineDeformableTransform.h"

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

namespace itk
{
/** \class RecursiveBSplineTransform
 * \brief A recursive implementation of the B-spline transform
 *
 * The class is templated coordinate representation type (float or double),
 * the space dimension and the spline order.
 *
 * \ingroup ITKTransform
 */

template <typename TScalarType = double, unsigned int NDimensions = 3, unsigned int VSplineOrder = 3>
class ITK_TEMPLATE_EXPORT RecursiveBSplineTransform
  : public AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>
{
public:
  /** Standard class typedefs. */
  using Self = RecursiveBSplineTransform;
  using Superclass = AdvancedBSplineDeformableTransform<TScalarType, NDimensions, VSplineOrder>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RecursiveBSplineTransform, AdvancedBSplineDeformableTransform);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** The BSpline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Standard scalar type for this class. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::ParametersValueType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::DerivativeType;
  using typename Superclass::JacobianType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;

  /** Parameters as SpaceDimension number of images. */
  using typename Superclass::PixelType;
  using typename Superclass::ImageType;
  using typename Superclass::ImagePointer;
  // using typename Superclass::CoefficientImageArray;

  /** Typedefs for specifying the extend to the grid. */
  using typename Superclass::RegionType;
  using typename Superclass::IndexType;
  using typename Superclass::SizeType;
  using typename Superclass::SpacingType;
  using typename Superclass::DirectionType;
  using typename Superclass::OriginType;
  using typename Superclass::GridOffsetType;
  using OffsetValueType = typename GridOffsetType::OffsetValueType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;
  using typename Superclass::MovingImageGradientType;
  using typename Superclass::MovingImageGradientValueType;

  /** Interpolation weights function type. */
  using typename Superclass::WeightsFunctionType;
  using typename Superclass::WeightsFunctionPointer;
  using typename Superclass::WeightsType;
  using typename Superclass::ContinuousIndexType;
  using typename Superclass::DerivativeWeightsFunctionType;
  using typename Superclass::DerivativeWeightsFunctionPointer;
  using typename Superclass::SODerivativeWeightsFunctionType;
  using typename Superclass::SODerivativeWeightsFunctionPointer;

  /** Parameter index array type. */
  using typename Superclass::ParameterIndexArrayType;

  using RecursiveBSplineWeightFunctionType = typename itk::
    RecursiveBSplineInterpolationWeightFunction<TScalarType, NDimensions, VSplineOrder>; // TODO: get rid of this and
                                                                                         // use the kernels directly.

  /** Compute point transformation. This one is commonly used.
   * It calls RecursiveBSplineTransformImplementation2::InterpolateTransformPoint
   * for a recursive implementation.
   */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType &       ipp,
              JacobianType &               j,
              NonZeroJacobianIndicesType & nonZeroJacobianIndices) const override;

  /** Compute the inner product of the Jacobian with the moving image gradient.
   * The Jacobian is (partially) constructed inside this function, but not returned.
   */
  void
  EvaluateJacobianWithImageGradientProduct(const InputPointType &          ipp,
                                           const MovingImageGradientType & movingImageGradient,
                                           DerivativeType &                imageJacobian,
                                           NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType & ipp, SpatialJacobianType & sj) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType & ipp, SpatialHessianType & sh) const override;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override;

protected:
  RecursiveBSplineTransform();
  ~RecursiveBSplineTransform() override = default;

  using typename Superclass::JacobianImageType;
  using typename Superclass::JacobianPixelType;

  typename RecursiveBSplineWeightFunctionType::Pointer m_RecursiveBSplineWeightFunction;

  /** Compute the nonzero Jacobian indices. */
  void
  ComputeNonZeroJacobianIndices(NonZeroJacobianIndicesType & nonZeroJacobianIndices,
                                const RegionType &           supportRegion) const override;

private:
  RecursiveBSplineTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkRecursiveBSplineTransform.hxx"
#endif

#endif /* itkRecursiveBSplineTransform_h */
