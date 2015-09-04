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
#ifndef __itkRecursiveBSplineInterpolateImageFunction_h
#define __itkRecursiveBSplineInterpolateImageFunction_h

#include "itkAdvancedInterpolateImageFunction.h"

#include <vector>
#include "itkImageLinearIteratorWithIndex.h"
#include "vnl/vnl_matrix.h"
#include "itkBSplineDecompositionImageFilter.h"
#include "itkConceptChecking.h"
#include "itkCovariantVector.h"
#include "itkBSplineInterpolationWeightFunction2.h"
#include "itkBSplineInterpolationDerivativeWeightFunction.h"
#include "itkBSplineInterpolationSecondOrderDerivativeWeightFunction.h"
#include "itkBSplineDerivativeKernelFunction2.h"

#include "itkRecursiveBSplineInterpolationWeightFunction.h"

namespace itk
{

/** \class RecursiveBSplineInterpolateImageFunction
 * \brief Evaluates the B-Spline interpolation of an image recursively.  Spline order may be from 0 to 5.
 *
 * This class defines N-Dimension B-Spline transformation, but in a recursive manner, such that it is much
 * faster than the itk class BSplineInterpolateImageFunction. This class is also templated over the spline order.
 * Inline classes are defined to obtain the B-spline weights for the chosen spline order. For backward compatibility
 * reasons a wrapper class was made, which is not templated over the spline order, called
 * RecursiveBSplineInterpolateImageFunctionWrapper.
 *
 * Computationally there are only round-off differences of order 1e-16 with the old ITK class, due to the fact that
 * floating point additions and multiplications are not commutative. Results are published in:
 *      W. Huizinga, S. Klein and D.H.J. Poot,
 *      "Fast Multidimensional B-spline Interpolation using Template Metaprogramming",
 *      Workshop on Biomedical Image Registration WBIR'14, July 2014, London.
 *
 * The B spline coefficients are calculated through the
 * BSplineDecompositionImageFilter
 *
 * Limitations:  Spline order must be between 0 and 5.
 *               Spline order must be set before setting the image.
 *               Uses mirror boundary conditions.
 *               Requires the same order of Spline for each dimension.
 *               Spline is determined in all dimensions, cannot selectively               <-- MS: do not understand this comment
 *                  pick dimension for calculating spline.
 */

template< class TImageType,
          class TCoordRep = double,
          class TCoefficientType = double,
          unsigned int SplineOrder = 3 >
class RecursiveBSplineInterpolateImageFunction :
  public AdvancedInterpolateImageFunction< TImageType, TCoordRep >
{
public:
  /** Standard class typedefs. */
  typedef RecursiveBSplineInterpolateImageFunction                  Self;
  typedef AdvancedInterpolateImageFunction< TImageType, TCoordRep > Superclass;
  typedef SmartPointer< Self >                                      Pointer;
  typedef SmartPointer< const Self >                                ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( RecursiveBSplineInterpolateImageFunction, AdvancedInterpolateImageFunction );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  /** InputImageType typedef support. */
  typedef typename Superclass::InputImageType InputImageType;

  /** Dimension underlying input image. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass::ImageDimension );

  /** Index typedef support. */
  typedef typename Superclass::IndexType IndexType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;

  /** PointType typedef support */
  typedef typename Superclass::PointType PointType;

  /** Iterator typedef support */
  typedef ImageLinearIteratorWithIndex< TImageType > Iterator;

  /** Internal Coefficient typedef support */
  typedef TCoefficientType CoefficientDataType;
  typedef Image< CoefficientDataType, itkGetStaticConstMacro( ImageDimension ) >   CoefficientImageType;

  /** Define filter for calculating the BSpline coefficients */
  typedef BSplineDecompositionImageFilter< TImageType, CoefficientImageType > CoefficientFilter;
  typedef typename CoefficientFilter::Pointer    CoefficientFilterPointer;

  /** Derivative typedef support */
  typedef typename Superclass::CovariantVectorType CovariantVectorType;
  typedef itk::Matrix< OutputType, ImageDimension, ImageDimension > MatrixType;

  /** Interpolation weights function type. */
  typedef BSplineInterpolationWeightFunction2< TCoordRep,ImageDimension, SplineOrder > WeightsFunctionType;
  typedef typename WeightsFunctionType::Pointer             WeightsFunctionPointer;
  typedef typename WeightsFunctionType::WeightsType         WeightsType;
  typedef BSplineInterpolationDerivativeWeightFunction< TCoordRep, ImageDimension, SplineOrder> DerivativeWeightsFunctionType;
  typedef typename DerivativeWeightsFunctionType::Pointer DerivativeWeightsFunctionPointer;
  typedef BSplineInterpolationSecondOrderDerivativeWeightFunction< TCoordRep, ImageDimension, SplineOrder > SODerivativeWeightsFunctionType;
  typedef typename SODerivativeWeightsFunctionType::Pointer SODerivativeWeightsFunctionPointer;

  /** Parameter index array type. */
  typedef typename itk::RecursiveBSplineInterpolationWeightFunction<
    TCoordRep, ImageDimension, SplineOrder >                      RecursiveBSplineWeightFunctionType;//TODO: get rid of this and use the kernels directly.

  /** Interpolation kernel type. */
  typedef BSplineKernelFunction2< SplineOrder > KernelType;
  typedef BSplineDerivativeKernelFunction2< SplineOrder > DerivativeKernelType;
  typedef BSplineSecondOrderDerivativeKernelFunction2< SplineOrder > SecondOrderDerivativeKernelType;

  /** Evaluate the function at a ContinuousIndex position.
   *
   * Returns the B-Spline interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method.
   */
  OutputType Evaluate( const PointType & point ) const;

  OutputType EvaluateAtContinuousIndex( const ContinuousIndexType & index ) const;

  CovariantVectorType EvaluateDerivative( const PointType & point) const;

  CovariantVectorType EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const;

  MatrixType EvaluateHessian( const PointType & point ) const;

  MatrixType EvaluateHessianAtContinuousIndex( const ContinuousIndexType & index ) const;


  void EvaluateValueAndDerivative(
    const PointType & point, OutputType & value, CovariantVectorType & deriv ) const;

  void EvaluateValueAndDerivativeAtContinuousIndex(
    const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const;

  void EvaluateValueAndDerivativeAndHessianAtContinuousIndex(
    const ContinuousIndexType & x, OutputType & value, CovariantVectorType & derivative, MatrixType & hessian) const;

  void EvaluateValueAndDerivativeAndHessian(
    const PointType & point, OutputType & value, CovariantVectorType & deriv, MatrixType & hessian) const;

  /** The UseImageDirection flag determines whether image derivatives are
   * computed with respect to the image grid or with respect to the physical
   * space. When this flag is ON the derivatives are computed with respect to
   * the coordinate system of physical space. The difference is whether we take
   * into account the image Direction or not. The flag ON will take into
   * account the image direction and will result in an extra matrix
   * multiplication compared to the amount of computation performed when the
   * flag is OFF.
   * The default value of this flag is On.
   */
  itkSetMacro( UseImageDirection, bool );
  itkGetConstMacro( UseImageDirection, bool );
  itkBooleanMacro( UseImageDirection );

  /** Set the input image, also in the coefficient filter. */
  void SetInputImage( const TImageType * inputData );

protected:
  /** Interpolation kernel. */
  typename KernelType::Pointer m_Kernel;
  typename DerivativeKernelType::Pointer m_DerivativeKernel;
  typename SecondOrderDerivativeKernelType::Pointer m_SecondOrderDerivativeKernel;

  RecursiveBSplineInterpolateImageFunction();
  ~RecursiveBSplineInterpolateImageFunction(){};

  void PrintSelf( std::ostream & os, Indent indent ) const;

  std::vector< CoefficientDataType >          m_Scratch;
  typename TImageType::SizeType               m_DataLength;
  typename CoefficientImageType::ConstPointer m_Coefficients;

  typename RecursiveBSplineWeightFunctionType::Pointer m_RecursiveBSplineWeightFunction;

private:
  RecursiveBSplineInterpolateImageFunction(const Self &); // purposely not implemented
  void operator=(const Self &);                           // purposely not implemented

  /** Determines the indices to use give the splines region of support. */
  void DetermineRegionOfSupport( vnl_matrix< long > & evaluateIndex,  const ContinuousIndexType & x ) const;

  /** Set the indices in evaluateIndex at the boundaries based on mirror
   * boundary conditions.
   */
  void ApplyMirrorBoundaryConditions( vnl_matrix< long > & evaluateIndex ) const;

  CoefficientFilterPointer m_CoefficientFilter;

  /** flag to take or not the image direction into account when computing the
   * derivatives.
   */
  bool m_UseImageDirection;

  /** Member variable containing the table with step sizes to take in each dimension
   * to obtain the correct index of the coefficient (vector alpha in the mentioned paper).
   */
  IndexType m_OffsetTable;

  typename InputImageType::SpacingType m_Spacing;
  InputImageType * m_inputImage;
};
} // namespace itk


#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRecursiveBSplineInterpolateImageFunction.hxx"
#endif

#endif //__itkRecursiveBSplineInterpolateImageFunction_h
