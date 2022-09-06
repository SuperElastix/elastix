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
#ifndef itkRecursiveBSplineInterpolationWeightFunction_h
#define itkRecursiveBSplineInterpolationWeightFunction_h

#include "itkBSplineInterpolationWeightFunction.h"

#include "itkBSplineKernelFunction2.h"
#include "itkBSplineDerivativeKernelFunction2.h"
#include "itkBSplineSecondOrderDerivativeKernelFunction2.h"
#include "itkMath.h"

namespace itk
{
/** \class RecursiveBSplineInterpolationWeightFunction
 * \brief Returns the weights over the support region used for B-spline
 * interpolation/reconstruction.
 *
 * Computes/evaluate the B-spline interpolation weights over the
 * support region of the B-spline.
 *
 * This class is templated over the coordinate representation type,
 * the space dimension and the spline order.
 *
 * \sa Point
 * \sa Index
 * \sa ContinuousIndex
 *
 * \ingroup Functions ImageInterpolators
 * \ingroup ITKCommon
 */
template <typename TCoordRep = float, unsigned int VSpaceDimension = 2, unsigned int VSplineOrder = 3>
class ITK_TEMPLATE_EXPORT RecursiveBSplineInterpolationWeightFunction
  : public BSplineInterpolationWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(RecursiveBSplineInterpolationWeightFunction);

  /** Standard class typedefs. */
  using Self = RecursiveBSplineInterpolationWeightFunction;
  using Superclass = BSplineInterpolationWeightFunction<TCoordRep, VSpaceDimension, VSplineOrder>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RecursiveBSplineInterpolationWeightFunction, FunctionBase);

  /** Space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, VSpaceDimension);

  /** Spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Typedefs from superclass*/
  using typename Superclass::WeightsType;
  using typename Superclass::IndexType;
  using typename Superclass::SizeType;
  using typename Superclass::ContinuousIndexType;

  /** The number of weights. */
  static constexpr unsigned NumberOfWeights = (VSplineOrder + 1) * VSpaceDimension;

  /** The number of indices. */
  static constexpr unsigned NumberOfIndices = Math::UnsignedPower(VSplineOrder + 1, VSpaceDimension);

  /** Get number of indices. */
  itkGetConstMacro(NumberOfIndices, unsigned int);

  WeightsType
  Evaluate(const ContinuousIndexType & index, IndexType & startIndex) const;

  WeightsType
  EvaluateDerivative(const ContinuousIndexType & index, const IndexType & startIndex) const;

  WeightsType
  EvaluateSecondOrderDerivative(const ContinuousIndexType & index, const IndexType & startIndex) const;

protected:
  RecursiveBSplineInterpolationWeightFunction();
  ~RecursiveBSplineInterpolationWeightFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Evaluate the weights at specified ContinousIndex position.
   * Subclasses must provide this method. */
  WeightsType
  Evaluate(const ContinuousIndexType & index) const override;

  /** Evaluate the weights at specified ContinousIndex position.
   * The weights are returned in the user specified container.
   * This function assume that weights can hold
   * (SplineOrder + 1)^(SpaceDimension) elements. For efficiency,
   * no size checking is done.
   * On return, startIndex contains the start index of the
   * support region over which the weights are defined.
   */
  void
  Evaluate(const ContinuousIndexType & index, WeightsType & weights, IndexType & startIndex) const override;

  /** Private members; We unfortunatly cannot use those of the superclass. */
  unsigned int m_NumberOfWeights;
  unsigned int m_NumberOfIndices;
  SizeType     m_SupportSize;

  /** Interpolation kernel type. */
  using KernelType = BSplineKernelFunction2<VSplineOrder>;
  using DerivativeKernelType = BSplineDerivativeKernelFunction2<VSplineOrder>;
  using SecondOrderDerivativeKernelType = BSplineSecondOrderDerivativeKernelFunction2<VSplineOrder>;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkRecursiveBSplineInterpolationWeightFunction.hxx"
#endif

#endif
