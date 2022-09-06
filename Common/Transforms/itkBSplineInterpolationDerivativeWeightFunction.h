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
#ifndef itkBSplineInterpolationDerivativeWeightFunction_h
#define itkBSplineInterpolationDerivativeWeightFunction_h

#include "itkBSplineInterpolationWeightFunctionBase.h"

namespace itk
{

/** \class BSplineInterpolationDerivativeWeightFunction
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
 */

template <class TCoordRep = float, unsigned int VSpaceDimension = 2, unsigned int VSplineOrder = 3>
class ITK_TEMPLATE_EXPORT BSplineInterpolationDerivativeWeightFunction
  : public BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineInterpolationDerivativeWeightFunction);

  /** Standard class typedefs. */
  using Self = BSplineInterpolationDerivativeWeightFunction;
  using Superclass = BSplineInterpolationWeightFunctionBase<TCoordRep, VSpaceDimension, VSplineOrder>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineInterpolationDerivativeWeightFunction, BSplineInterpolationWeightFunctionBase);

  /** Space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, VSpaceDimension);

  /** Spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Typedefs from Superclass. */
  using typename Superclass::WeightsType;
  using typename Superclass::IndexType;
  using typename Superclass::SizeType;
  using typename Superclass::ContinuousIndexType;

  /** Set the first order derivative direction. */
  virtual void
  SetDerivativeDirection(unsigned int dir);

protected:
  BSplineInterpolationDerivativeWeightFunction();
  ~BSplineInterpolationDerivativeWeightFunction() override = default;

  /** Interpolation kernel types. */
  using typename Superclass::KernelType;
  using typename Superclass::DerivativeKernelType;
  using typename Superclass::SecondOrderDerivativeKernelType;
  using typename Superclass::TableType;
  using typename Superclass::OneDWeightsType;

  /** Compute the 1D weights, which are:
   * \f[ \beta( x[i] - startIndex[i] ), \beta( x[i] - startIndex[i] - 1 ),
   * \beta( x[i] - startIndex[i] - 2 ), \beta( x[i] - startIndex[i] - 3 ) \f],
   * with \f$\beta( x ) = \beta^2( x + 1/2 ) - \beta^2( x - 1/2 )\f$, in case of the
   * derivative direction, and just \f$\beta(x) = \beta^3(x)\f$ for the non-derivative
   * directions.
   */
  void
  Compute1DWeights(const ContinuousIndexType & index,
                   const IndexType &           startIndex,
                   OneDWeightsType &           weights1D) const override;

  /** Print the member variables. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  /** Member variables. */
  unsigned int m_DerivativeDirection;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBSplineInterpolationDerivativeWeightFunction.hxx"
#endif

#endif
