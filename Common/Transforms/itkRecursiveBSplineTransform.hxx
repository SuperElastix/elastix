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
#ifndef itkRecursiveBSplineTransform_hxx
#define itkRecursiveBSplineTransform_hxx

#include "itkRecursiveBSplineTransform.h"


namespace itk
{

/**
 * ********************* TransformPoint ****************************
 */

template <typename TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
auto
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::TransformPoint(const InputPointType & point) const
  -> OutputPointType
{
  /** Initialize output point. */
  OutputPointType outputPoint;

  /** Allocate weights on the stack: */

  /** Check if the coefficient image has been set. */
  if (!this->m_CoefficientImages[0])
  {
    itkWarningMacro(<< "B-spline coefficients have not been set");
    outputPoint = point;
    return outputPoint;
  }

  /** Convert to continuous index. */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(point);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and return the input point
  bool inside = this->InsideValidRegion(cindex);
  if (!inside)
  {
    outputPoint = point;
    return outputPoint;
  }

  // Compute interpolation weighs and store them in weights1D
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);

  /** Initialize (helper) variables. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[0]->GetOffsetTable();
  OffsetValueType         totalOffsetToSupportIndex = 0;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    totalOffsetToSupportIndex += supportIndex[j] * bsplineOffsetTable[j];
  }

  ScalarType * mu[SpaceDimension];
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    mu[j] = this->m_CoefficientImages[j]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Call the recursive TransformPoint function. */
  ScalarType displacement[SpaceDimension];
  ImplementationType::TransformPoint(displacement, mu, bsplineOffsetTable, weights1D.data());

  // The output point is the start point + displacement.
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    outputPoint[j] = displacement[j] + point[j];
  }

  return outputPoint;
} // end TransformPoint()


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetJacobian(
  const InputPointType &       inputPoint,
  JacobianType &               jacobian,
  NonZeroJacobianIndicesType & nonZeroJacobianIndices) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  /** Initialize. */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if ((jacobian.cols() != nnzji) || (jacobian.rows() != SpaceDimension))
  {
    jacobian.SetSize(SpaceDimension, nnzji);
    jacobian.Fill(0.0);
  }

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  if (!this->InsideValidRegion(cindex))
  {
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);

  /** Recursively compute the first numberOfIndices entries of the Jacobian.
   * They are directly written in the Jacobian matrix memory block.
   * The pointer has changed after this function call.
   */
  ParametersValueType * jacobianPointer = jacobian.data_block();
  ImplementationType::GetJacobian(jacobianPointer, weights1D.data(), 1.0);

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobian()


/**
 * ********************* EvaluateJacobianAndImageGradientProduct ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::EvaluateJacobianWithImageGradientProduct(
  const InputPointType &          inputPoint,
  const MovingImageGradientType & movingImageGradient,
  DerivativeType &                imageJacobian,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  /** NOTE: if the support region does not lie totally within the grid
   * we assume zero displacement and zero Jacobian.
   */
  const NumberOfParametersType nnzji = this->GetNumberOfNonZeroJacobianIndices();
  if (!this->InsideValidRegion(cindex))
  {
    nonZeroJacobianIndices.resize(nnzji);
    for (NumberOfParametersType i = 0; i < nnzji; ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);

  /** Recursively compute the inner product of the Jacobian and the moving image gradient.
   * The pointer has changed after this function call.
   */
  // ParametersValueType migArray[ SpaceDimension ];
  double migArray[SpaceDimension]; // InternalFloatType
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    migArray[j] = movingImageGradient[j];
  }
  ParametersValueType * imageJacobianPointer = imageJacobian.data_block();
  ImplementationType::EvaluateJacobianWithImageGradientProduct(imageJacobianPointer, migArray, weights1D.data(), 1.0);

  /** Setup support region needed for the nonZeroJacobianIndices. */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Compute the nonzero Jacobian indices.
   * Takes a significant portion of the computation time of this function.
   */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end EvaluateJacobianWithImageGradientProduct()


/**
 * ********************* GetSpatialJacobian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetSpatialJacobian(const InputPointType & inputPoint,
                                                                                  SpatialJacobianType &  sj) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity spatial Jacobian
  if (!this->InsideValidRegion(cindex))
  {
    sj.SetIdentity();
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);
  const WeightsType derivativeWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateDerivative(cindex, supportIndex);

  /** Compute the offset to the start index. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[0]->GetOffsetTable();
  OffsetValueType         totalOffsetToSupportIndex = 0;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    totalOffsetToSupportIndex += supportIndex[j] * bsplineOffsetTable[j];
  }

  /** Get handles to the mu's. */
  ScalarType * mu[SpaceDimension];
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    mu[j] = this->m_CoefficientImages[j]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Recursively compute the spatial Jacobian. */
  double spatialJacobian[SpaceDimension * (SpaceDimension + 1)]; // double
  ImplementationType::GetSpatialJacobian(
    spatialJacobian, mu, bsplineOffsetTable, weights1D.data(), derivativeWeights1D.data());

  /** Copy the correct elements to the spatial Jacobian.
   * The first SpaceDimension elements are actually the displacement, i.e. the recursive
   * function GetSpatialJacobian() has the TransformPoint as a free by-product.
   */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j < SpaceDimension; ++j)
    {
      sj(i, j) = spatialJacobian[i + (j + 1) * SpaceDimension];
    }
  }

  /** Take into account grid spacing and direction cosines. */
  sj = sj * this->m_PointToIndexMatrix2;

  /** Add the identity matrix, as this is a transformation, not displacement. */
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    sj(j, j) += 1.0;
  }

} // end GetSpatialJacobian()


/**
 * ********************* GetSpatialHessian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetSpatialHessian(const InputPointType & inputPoint,
                                                                                 SpatialHessianType &   sh) const
{
  /** Convert the physical point to a continuous index, which
   * is needed for the evaluate functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero spatial Hessian
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < sh.Size(); ++i)
    {
      sh[i].Fill(0.0);
    }
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);
  const WeightsType derivativeWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateDerivative(cindex, supportIndex);
  const WeightsType hessianWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateSecondOrderDerivative(cindex, supportIndex);

  /** Compute the offset to the start index. */
  const OffsetValueType * bsplineOffsetTable = this->m_CoefficientImages[0]->GetOffsetTable();
  OffsetValueType         totalOffsetToSupportIndex = 0;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    totalOffsetToSupportIndex += supportIndex[j] * bsplineOffsetTable[j];
  }

  /** Get handles to the mu's. */
  ScalarType * mu[SpaceDimension];
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    mu[j] = this->m_CoefficientImages[j]->GetBufferPointer() + totalOffsetToSupportIndex;
  }

  /** Recursively compute the spatial Hessian. */
  double spatialHessian[SpaceDimension * (SpaceDimension + 1) * (SpaceDimension + 2) / 2];
  ImplementationType::GetSpatialHessian(
    spatialHessian, mu, bsplineOffsetTable, weights1D.data(), derivativeWeights1D.data(), hessianWeights1D.data());

  /** Copy the correct elements to the spatial Hessian.
   * The first SpaceDimension elements are actually the displacement, i.e. the recursive
   * function GetSpatialHessian() has the TransformPoint as a free by-product.
   * In addition, the spatial Jacobian is a by-product.
   */
  unsigned int k = 2 * SpaceDimension;
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j < (i + 1) * SpaceDimension; ++j)
    {
      sh[j % SpaceDimension](i, j / SpaceDimension) = spatialHessian[k + j];
    }
    k += (i + 2) * SpaceDimension;
  }

  /** Mirror, as only the lower triangle is now filled. */
  for (unsigned int i = 0; i < SpaceDimension; ++i)
  {
    for (unsigned int j = 0; j < SpaceDimension - 1; ++j)
    {
      for (unsigned int k = 1; k < SpaceDimension; ++k)
      {
        sh[i](j, k) = sh[i](k, j);
      }
    }
  }

  /** Take into account grid spacing and direction matrix. */
  for (unsigned int dim = 0; dim < SpaceDimension; ++dim)
  {
    sh[dim] = this->m_PointToIndexMatrixTransposed2 * (sh[dim] * this->m_PointToIndexMatrix2);
  }

} // end GetSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsj.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < jsj.size(); ++i)
    {
      jsj[i].Fill(0.0);
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }

  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);
  const WeightsType derivativeWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateDerivative(cindex, supportIndex);

  /** Allocate memory for jsj. If you want also the Jacobian,
   * numberOfIndices more elements are needed.
   */
  const double dummy[1] = { 1.0 };

  /** Recursively expand all weights (destroys dummy), and multiply with dc. */
  const double * dc = this->m_PointToIndexMatrix2.GetVnlMatrix().data_block();
  double *       jsjPtr2 = jsj[0].GetVnlMatrix().data_block();
  ImplementationType::GetJacobianOfSpatialJacobian(jsjPtr2, weights1D.data(), derivativeWeights1D.data(), dc, dummy);

  /** Setup support region needed for the nonZeroJacobianIndices. */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialJacobian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetJacobianOfSpatialJacobian(
  const InputPointType &          inputPoint,
  SpatialJacobianType &           sj,
  JacobianOfSpatialJacobianType & jsj,
  NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const
{
  this->GetJacobianOfSpatialJacobian(inputPoint, jsj, nonZeroJacobianIndices);
  this->GetSpatialJacobian(inputPoint, sj);
} // end GetJacobianOfSpatialJacobian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  // Can only compute Jacobian if parameters are set via
  // SetParameters or SetParametersByValue
  if (this->m_InputParametersPointer == nullptr)
  {
    itkExceptionMacro(<< "Cannot compute Jacobian: parameters not set");
  }

  jsh.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Convert the physical point to a continuous index, which
   * is needed for the 'Evaluate()' functions below.
   */
  const ContinuousIndexType cindex = this->TransformPointToContinuousGridIndex(inputPoint);

  // NOTE: if the support region does not lie totally within the grid
  // we assume zero displacement and identity sj and zero jsj.
  if (!this->InsideValidRegion(cindex))
  {
    for (unsigned int i = 0; i < jsh.size(); ++i)
    {
      for (unsigned int j = 0; j < jsh[i].Size(); ++j)
      {
        jsh[i][j].Fill(0.0);
      }
    }
    nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());
    for (NumberOfParametersType i = 0; i < this->GetNumberOfNonZeroJacobianIndices(); ++i)
    {
      nonZeroJacobianIndices[i] = i;
    }
    return;
  }


  /** Compute the interpolation weights.
   * In contrast to the normal B-spline weights function, the recursive version
   * returns the individual weights instead of the multiplied ones.
   */
  IndexType         supportIndex;
  const WeightsType weights1D = this->m_RecursiveBSplineWeightFunction.Evaluate(cindex, supportIndex);
  const WeightsType derivativeWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateDerivative(cindex, supportIndex);
  const WeightsType hessianWeights1D =
    this->m_RecursiveBSplineWeightFunction.EvaluateSecondOrderDerivative(cindex, supportIndex);

  /** Recursively expand all weights (destroys dummy and jshPtr points to last element afterwards).
   * This version also performs pre- and post-multiplication with the matrices dc^T and dc, respectively.
   * Other differences are that the complete matrix is returned, not just the upper triangle.
   * And the results are directly written to the final jsh, avoiding an additional copy.
   */
  double *       jshPtr = jsh[0][0].GetVnlMatrix().data_block();
  const double * dc = this->m_PointToIndexMatrix2.GetVnlMatrix().data_block();
  const double   dummy[1] = { 1.0 };
  ImplementationType::GetJacobianOfSpatialHessian(
    jshPtr, weights1D.data(), derivativeWeights1D.data(), hessianWeights1D.data(), dc, dummy);

  /** Setup support region needed for the nonZeroJacobianIndices. */
  const RegionType supportRegion(supportIndex, Superclass::m_SupportSize);

  /** Compute the nonzero Jacobian indices. */
  this->ComputeNonZeroJacobianIndices(nonZeroJacobianIndices, supportRegion);

} // end GetJacobianOfSpatialHessian()


/**
 * ********************* GetJacobianOfSpatialHessian ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::GetJacobianOfSpatialHessian(
  const InputPointType &         inputPoint,
  SpatialHessianType &           sh,
  JacobianOfSpatialHessianType & jsh,
  NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const
{
  this->GetJacobianOfSpatialHessian(inputPoint, jsh, nonZeroJacobianIndices);
  this->GetSpatialHessian(inputPoint, sh);
} // end GetJacobianOfSpatialHessian()


/**
 * ********************* ComputeNonZeroJacobianIndices ****************************
 */

template <class TScalar, unsigned int NDimensions, unsigned int VSplineOrder>
void
RecursiveBSplineTransform<TScalar, NDimensions, VSplineOrder>::ComputeNonZeroJacobianIndices(
  NonZeroJacobianIndicesType & nonZeroJacobianIndices,
  const RegionType &           supportRegion) const
{
  /** Initialize some helper variables. */
  const unsigned long parametersPerDim = this->GetNumberOfParametersPerDimension();
  nonZeroJacobianIndices.resize(this->GetNumberOfNonZeroJacobianIndices());

  /** Compute total offset at start index. */
  const IndexType         startIndex = supportRegion.GetIndex();
  const OffsetValueType * gridOffsetTable = this->m_CoefficientImages[0]->GetOffsetTable();
  OffsetValueType         totalOffsetToSupportIndex = 0;
  for (unsigned int j = 0; j < SpaceDimension; ++j)
  {
    totalOffsetToSupportIndex += startIndex[j] * gridOffsetTable[j];
  }

  /** Call the recursive implementation. */
  unsigned long   currentIndex = totalOffsetToSupportIndex;
  unsigned long * nzjiPointer = &nonZeroJacobianIndices[0];
  ImplementationType::ComputeNonZeroJacobianIndices(nzjiPointer, parametersPerDim, currentIndex, gridOffsetTable);

} // end ComputeNonZeroJacobianIndices()


} // end namespace itk

#endif
