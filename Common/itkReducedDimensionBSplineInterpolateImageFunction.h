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
  Module:    $RCSfile: itkReducedDimBSplineInterpolateImageFunction.h,v $
  Language:  C++
  Date:      $Date: 2009-04-25 12:27:05 $
  Version:   $Revision: 1.24 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkReducedDimensionBSplineInterpolateImageFunction_h
#define itkReducedDimensionBSplineInterpolateImageFunction_h

#include <vector>

#include "itkInterpolateImageFunction.h"
#include <vnl/vnl_matrix.h>

#include "itkMultiOrderBSplineDecompositionImageFilter.h"
#include "itkConceptChecking.h"
#include "itkCovariantVector.h"

namespace itk
{
/** \class ReducedDimensionBSplineInterpolateImageFunction
 * \brief Evaluates the B-Spline interpolation of an image.  Spline order may be from 0 to 5.
 *
 * This class defines N-Dimension B-Spline transformation.
 * It is based on:\n
 *    [1] M. Unser,
 *       "Splines: A Perfect Fit for Signal and Image Processing,"
 *        IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38,
 *        November 1999.\n
 *    [2] M. Unser, A. Aldroubi and M. Eden,
 *        "B-Spline Signal Processing: Part I--Theory,"
 *        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821-832,
 *        February 1993.\n
 *    [3] M. Unser, A. Aldroubi and M. Eden,
 *        "B-Spline Signal Processing: Part II--Efficient Design and Applications,"
 *        IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834-848,
 *        February 1993.\n
 * And code obtained from bigwww.epfl.ch by Philippe Thevenaz.
 *
 * The B spline coefficients are calculated through the
 * MultiOrderBSplineDecompositionImageFilter to enable a zero-th order
 * for the last dimension.
 *
 * Limitations:  Spline order must be between 0 and 5.
 *               Spline order must be set before setting the image.
 *               Requires same spline order for every dimension.
 *               Uses mirror boundary conditions.
 *               Spline is determined in all dimensions, cannot selectively
 *                  pick dimension for calculating spline.
 *
 * \sa MultiOrderBSplineDecompositionImageFilter
 *
 * \ingroup ImageFunctions
 */
template <class TImageType, class TCoordRep = double, class TCoefficientType = double>
class ITK_TEMPLATE_EXPORT ReducedDimensionBSplineInterpolateImageFunction
  : public InterpolateImageFunction<TImageType, TCoordRep>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ReducedDimensionBSplineInterpolateImageFunction);

  /** Standard class typedefs. */
  using Self = ReducedDimensionBSplineInterpolateImageFunction;
  using Superclass = InterpolateImageFunction<TImageType, TCoordRep>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ReducedDimensionBSplineInterpolateImageFunction, InterpolateImageFunction);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  using typename Superclass::OutputType;

  using SizeType = typename TImageType::SizeType;

  /** InputImageType typedef support. */
  using typename Superclass::InputImageType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

  /** Index typedef support. */
  using typename Superclass::IndexType;

  /** ContinuousIndex typedef support. */
  using typename Superclass::ContinuousIndexType;

  /** PointType typedef support */
  using typename Superclass::PointType;

  /** Internal Coefficient typedef support */
  using CoefficientDataType = TCoefficientType;
  using CoefficientImageType = Image<CoefficientDataType, Self::ImageDimension>;

  /** Define filter for calculating the BSpline coefficients */
  using CoefficientFilter = MultiOrderBSplineDecompositionImageFilter<TImageType, CoefficientImageType>;

  using CoefficientFilterPointer = typename CoefficientFilter::Pointer;

  /** Evaluate the function at a ContinuousIndex position.
   *
   * Returns the B-Spline interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override;

  /** Derivative typedef support */
  using CovariantVectorType = CovariantVector<OutputType, Self::ImageDimension>;

  CovariantVectorType
  EvaluateDerivative(const PointType & point) const
  {
    const auto index = this->GetInputImage()->template TransformPhysicalPointToContinuousIndex<TCoordRep>(point);
    return (this->EvaluateDerivativeAtContinuousIndex(index));
  }


  CovariantVectorType
  EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const;

  /** Get/Sets the Spline Order, supports 0th - 5th order splines. The default
   *  is a 3rd order spline. */
  void
  SetSplineOrder(unsigned int SplineOrder);

  itkGetConstMacro(SplineOrder, int);

  /** Set the input image.  This must be set by the user. */
  void
  SetInputImage(const TImageType * inputData) override;

  /** The UseImageDirection flag determines whether image derivatives are
   * computed with respect to the image grid or with respect to the physical
   * space. When this flag is ON the derivatives are computed with respect to
   * the coordinate system of physical space. The difference is whether we take
   * into account the image Direction or not. The flag ON will take into
   * account the image direction and will result in an extra matrix
   * multiplication compared to the amount of computation performed when the
   * flag is OFF.
   * The default value of this flag is the same as the CMAKE option
   * ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE (i.e ON by default when ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE is ON,
   * and  OFF by default when ITK_IMAGE_BEHAVES_AS_ORIENTED_IMAGE is
   * OFF). */
  itkSetMacro(UseImageDirection, bool);
  itkGetConstMacro(UseImageDirection, bool);
  itkBooleanMacro(UseImageDirection);

protected:
  ReducedDimensionBSplineInterpolateImageFunction();
  ~ReducedDimensionBSplineInterpolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  // These are needed by the smoothing spline routine.
  typename TImageType::SizeType m_DataLength;  // Image size
  unsigned int                  m_SplineOrder; // User specified spline order (3rd or cubic is the default)

  typename CoefficientImageType::ConstPointer m_Coefficients; // Spline coefficients

private:
  SizeType
  GetRadius() const override
  {
    return SizeType::Filled(m_SplineOrder + 1);
  }

  /** Determines the weights for interpolation of the value x */
  void
  SetInterpolationWeights(const ContinuousIndexType & x,
                          const vnl_matrix<long> &    EvaluateIndex,
                          vnl_matrix<double> &        weights,
                          unsigned int                splineOrder) const;

  /** Determines the weights for the derivative portion of the value x */
  void
  SetDerivativeWeights(const ContinuousIndexType & x,
                       const vnl_matrix<long> &    EvaluateIndex,
                       vnl_matrix<double> &        weights,
                       unsigned int                splineOrder) const;

  /** Precomputation for converting the 1D index of the interpolation neighborhood
   * to an N-dimensional index. */
  void
  GeneratePointsToIndex();

  /** Determines the indicies to use give the splines region of support */
  void
  DetermineRegionOfSupport(vnl_matrix<long> &          evaluateIndex,
                           const ContinuousIndexType & x,
                           unsigned int                splineOrder) const;

  /** Set the indicies in evaluateIndex at the boundaries based on mirror
   * boundary conditions. */
  void
  ApplyMirrorBoundaryConditions(vnl_matrix<long> & evaluateIndex, unsigned int splineOrder) const;

  std::vector<IndexType> m_PointsToIndex; // Preallocation of interpolation neighborhood indicies

  CoefficientFilterPointer m_CoefficientFilter;

  // flag to take or not the image direction into account when computing the
  // derivatives.
  bool m_UseImageDirection;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkReducedDimensionBSplineInterpolateImageFunction.hxx"
#endif

#endif
