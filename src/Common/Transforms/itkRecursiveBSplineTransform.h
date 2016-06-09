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
#ifndef __itkRecursiveBSplineTransform_h
#define __itkRecursiveBSplineTransform_h

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

template< typename TScalarType = double,
  unsigned int NDimensions       = 3,
  unsigned int VSplineOrder      = 3 >
class RecursiveBSplineTransform :
  public AdvancedBSplineDeformableTransform< TScalarType, NDimensions, VSplineOrder >
{
public:

  /** Standard class typedefs. */
  typedef RecursiveBSplineTransform          Self;
  typedef AdvancedBSplineDeformableTransform<
    TScalarType, NDimensions, VSplineOrder > Superclass;
  typedef SmartPointer< Self >               Pointer;
  typedef SmartPointer< const Self >         ConstPointer;

  /** New macro for creation of through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineTransform, AdvancedBSplineDeformableTransform );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** The BSpline order. */
  itkStaticConstMacro( SplineOrder, unsigned int, VSplineOrder );

  /** Standard scalar type for this class. */
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::ParametersValueType       ParametersValueType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::DerivativeType            DerivativeType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;

  /** Parameters as SpaceDimension number of images. */
  typedef typename Superclass::PixelType    PixelType;
  typedef typename Superclass::ImageType    ImageType;
  typedef typename Superclass::ImagePointer ImagePointer;
  //typedef typename Superclass::CoefficientImageArray CoefficientImageArray;

  /** Typedefs for specifying the extend to the grid. */
  typedef typename Superclass::RegionType          RegionType;
  typedef typename Superclass::IndexType           IndexType;
  typedef typename Superclass::SizeType            SizeType;
  typedef typename Superclass::SpacingType         SpacingType;
  typedef typename Superclass::DirectionType       DirectionType;
  typedef typename Superclass::OriginType          OriginType;
  typedef typename Superclass::GridOffsetType      GridOffsetType;
  typedef typename GridOffsetType::OffsetValueType OffsetValueType;

  typedef typename Superclass::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType            InternalMatrixType;
  typedef typename Superclass::MovingImageGradientType       MovingImageGradientType;
  typedef typename Superclass::MovingImageGradientValueType  MovingImageGradientValueType;

  /** Interpolation weights function type. */
  typedef typename Superclass::WeightsFunctionType                WeightsFunctionType;
  typedef typename Superclass::WeightsFunctionPointer             WeightsFunctionPointer;
  typedef typename Superclass::WeightsType                        WeightsType;
  typedef typename Superclass::ContinuousIndexType                ContinuousIndexType;
  typedef typename Superclass::DerivativeWeightsFunctionType      DerivativeWeightsFunctionType;
  typedef typename Superclass::DerivativeWeightsFunctionPointer   DerivativeWeightsFunctionPointer;
  typedef typename Superclass::SODerivativeWeightsFunctionType    SODerivativeWeightsFunctionType;
  typedef typename Superclass::SODerivativeWeightsFunctionPointer SODerivativeWeightsFunctionPointer;

  /** Parameter index array type. */
  typedef typename Superclass::ParameterIndexArrayType ParameterIndexArrayType;

  typedef typename itk::RecursiveBSplineInterpolationWeightFunction<
    TScalarType, NDimensions, VSplineOrder >                      RecursiveBSplineWeightFunctionType; //TODO: get rid of this and use the kernels directly.

  /** Interpolation kernel type. */
  typedef BSplineKernelFunction2< itkGetStaticConstMacro( SplineOrder ) >                      KernelType;
  typedef BSplineDerivativeKernelFunction2< itkGetStaticConstMacro( SplineOrder ) >            DerivativeKernelType;
  typedef BSplineSecondOrderDerivativeKernelFunction2< itkGetStaticConstMacro( SplineOrder ) > SecondOrderDerivativeKernelType;

  /** Interpolation kernel. */
  typename KernelType::Pointer m_Kernel;
  typename DerivativeKernelType::Pointer m_DerivativeKernel;
  typename SecondOrderDerivativeKernelType::Pointer m_SecondOrderDerivativeKernel;

  /** Compute point transformation. This one is commonly used.
   * It calls RecursiveBSplineTransformImplementation2::InterpolateTransformPoint
   * for a recursive implementation.
   */
  virtual OutputPointType TransformPoint( const InputPointType & point ) const;

  /** Temporary function to test performance of functional recursive TransformPoint. */
  typedef ScalarType ** CoefficientPointerVectorType;
  typedef ScalarType *  OutputPointType2;

  OutputPointType TransformPointFunctionalRecursive( const InputPointType & point ) const;

  void TransformPointFunctionalRecursiveFunction( OutputPointType2 displacement, const CoefficientPointerVectorType mu,
    const OffsetValueType * gridOffsetTable, const double * weights1D, unsigned int D ) const;

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType & ipp,
    JacobianType & j,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the inner product of the Jacobian with the moving image gradient.
   * The Jacobian is (partially) constructed inside this function, but not returned.
   */
  virtual void EvaluateJacobianWithImageGradientProduct(
    const InputPointType & ipp,
    const MovingImageGradientType & movingImageGradient,
    DerivativeType & imageJacobian,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the spatial Jacobian of the transformation. */
  virtual void GetSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj ) const;

  /** Compute the spatial Hessian of the transformation. */
  virtual void GetSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh ) const;

  /** Compute the Jacobian of the spatial Jacobian of the transformation. */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Jacobian and the Jacobian of the
   * spatial Jacobian of the transformation.
   */
  virtual void GetJacobianOfSpatialJacobian(
    const InputPointType & ipp,
    SpatialJacobianType & sj,
    JacobianOfSpatialJacobianType & jsj,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute the Jacobian of the spatial Hessian of the transformation. */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

  /** Compute both the spatial Hessian and the Jacobian of the
   * spatial Hessian of the transformation.
   */
  virtual void GetJacobianOfSpatialHessian(
    const InputPointType & ipp,
    SpatialHessianType & sh,
    JacobianOfSpatialHessianType & jsh,
    NonZeroJacobianIndicesType & nonZeroJacobianIndices ) const;

protected:

  RecursiveBSplineTransform();
  virtual ~RecursiveBSplineTransform(){}

  typedef typename Superclass::JacobianImageType JacobianImageType;
  typedef typename Superclass::JacobianPixelType JacobianPixelType;

  typename RecursiveBSplineWeightFunctionType::Pointer m_RecursiveBSplineWeightFunction;

  /** Compute the nonzero Jacobian indices. */
  virtual void ComputeNonZeroJacobianIndices(
    NonZeroJacobianIndicesType & nonZeroJacobianIndices,
    const RegionType & supportRegion ) const;

private:

  RecursiveBSplineTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineTransform.hxx"
#endif

#endif /* __itkRecursiveBSplineTransform_h */
