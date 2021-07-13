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
#ifndef itkBSplineInterpolationWeightFunctionBase_h
#define itkBSplineInterpolationWeightFunctionBase_h

#include "itkFunctionBase.h"
#include "itkContinuousIndex.h"
#include "itkArray.h"
#include "itkArray2D.h"
#include "itkMatrix.h"
#include "itkBSplineKernelFunction2.h"
#include "itkBSplineDerivativeKernelFunction.h"
#include "itkBSplineSecondOrderDerivativeKernelFunction2.h"

namespace itk
{

/** Recursive template to retrieve the number of Bspline weights at compile time. */
template <unsigned int SplineOrder, unsigned int Dimension>
class ITK_TEMPLATE_EXPORT GetConstNumberOfWeightsHack
{
public:
  typedef GetConstNumberOfWeightsHack<SplineOrder, Dimension - 1> OneDimensionLess;
  itkStaticConstMacro(Value, unsigned long, (SplineOrder + 1) * OneDimensionLess::Value);
};

/** Partial template specialization to terminate the recursive loop. */
template <unsigned int SplineOrder>
class ITK_TEMPLATE_EXPORT GetConstNumberOfWeightsHack<SplineOrder, 0>
{
public:
  itkStaticConstMacro(Value, unsigned long, 1);
};

/** \class BSplineInterpolationWeightFunctionBase
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
class ITK_TEMPLATE_EXPORT BSplineInterpolationWeightFunctionBase
  : public FunctionBase<ContinuousIndex<TCoordRep, VSpaceDimension>, Array<double>>
{
public:
  /** Standard class typedefs. */
  typedef BSplineInterpolationWeightFunctionBase                                   Self;
  typedef FunctionBase<ContinuousIndex<TCoordRep, VSpaceDimension>, Array<double>> Superclass;
  typedef SmartPointer<Self>                                                       Pointer;
  typedef SmartPointer<const Self>                                                 ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineInterpolationWeightFunctionBase, FunctionBase);

  /** Space dimension. */
  itkStaticConstMacro(SpaceDimension, unsigned int, VSpaceDimension);

  /** Spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** The number of weights as a static const. */
  typedef GetConstNumberOfWeightsHack<itkGetStaticConstMacro(SplineOrder), itkGetStaticConstMacro(SpaceDimension)>
    GetConstNumberOfWeightsHackType;
  itkStaticConstMacro(NumberOfWeights, unsigned long, GetConstNumberOfWeightsHackType::Value);

  /** OutputType typedef support. */
  typedef Array<double> WeightsType;

  /** Index and size typedef support. */
  typedef Index<VSpaceDimension> IndexType;
  typedef Size<VSpaceDimension>  SizeType;

  /** ContinuousIndex typedef support. */
  typedef ContinuousIndex<TCoordRep, VSpaceDimension> ContinuousIndexType;

  /** Evaluate the weights at specified ContinousIndex position. */
  WeightsType
  Evaluate(const ContinuousIndexType & index) const override;

  /** Evaluate the weights at specified ContinousIndex position.
   * The weights are returned in the user specified container.
   * This function assume that the weights has a correct size. For efficiency,
   * no size checking is done.
   * On return, startIndex contains the start index of the
   * support region over which the weights are defined.
   */
  virtual void
  Evaluate(const ContinuousIndexType & cindex, const IndexType & startIndex, WeightsType & weights) const;

  /** Compute the start index of the support region. */
  void
  ComputeStartIndex(const ContinuousIndexType & index, IndexType & startIndex) const;

  /** Get support region size. */
  itkGetConstReferenceMacro(SupportSize, SizeType);

  /** Get number of weights. */
  itkGetConstMacro(NumberOfWeights, unsigned long);

protected:
  BSplineInterpolationWeightFunctionBase();
  ~BSplineInterpolationWeightFunctionBase() override = default;

  /** Interpolation kernel types. */
  typedef BSplineKernelFunction2<Self::SplineOrder>                      KernelType;
  typedef typename KernelType::Pointer                                   KernelPointer;
  typedef BSplineDerivativeKernelFunction<Self::SplineOrder>             DerivativeKernelType;
  typedef typename DerivativeKernelType::Pointer                         DerivativeKernelPointer;
  typedef BSplineSecondOrderDerivativeKernelFunction2<Self::SplineOrder> SecondOrderDerivativeKernelType;
  typedef typename SecondOrderDerivativeKernelType::Pointer              SecondOrderDerivativeKernelPointer;
  typedef typename KernelType::WeightArrayType                           WeightArrayType;

  /** Lookup table type. */
  typedef Array2D<unsigned long> TableType;

  /** Typedef for intermediary 1D weights.
   * The Matrix is at least twice as fast as std::vector< vnl_vector< double > >,
   * probably because of the fixed size at compile time.
   */
  typedef Matrix<double, itkGetStaticConstMacro(SpaceDimension), itkGetStaticConstMacro(SplineOrder) + 1>
    OneDWeightsType;

  /** Compute the 1D weights. */
  virtual void
  Compute1DWeights(const ContinuousIndexType & index,
                   const IndexType &           startIndex,
                   OneDWeightsType &           weights1D) const = 0;

  /** Print the member variables. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  unsigned long m_NumberOfWeights;
  SizeType      m_SupportSize;
  TableType     m_OffsetToIndexTable;

  /** Interpolation kernels. */
  typename KernelType::Pointer                      m_Kernel;
  typename DerivativeKernelType::Pointer            m_DerivativeKernel;
  typename SecondOrderDerivativeKernelType::Pointer m_SecondOrderDerivativeKernel;

private:
  BSplineInterpolationWeightFunctionBase(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** Function to initialize the support region. */
  void
  InitializeSupport(void);

  /** Function to initialize the offset table.
   * The offset table is a convenience table, just to
   * keep track where is what.
   */
  void
  InitializeOffsetToIndexTable(void);
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkBSplineInterpolationWeightFunctionBase.hxx"
#endif

#endif
