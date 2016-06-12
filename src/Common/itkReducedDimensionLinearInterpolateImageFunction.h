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
#ifndef __itkReducedDimensionLinearInterpolateImageFunction_h
#define __itkReducedDimensionLinearInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"

namespace itk
{
/** \class ReducedDimensionLinearInterpolateImageFunction
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
template< class TImageType,class TCoordRep = double >
class ReducedDimensionLinearInterpolateImageFunction :
  public InterpolateImageFunction< TImageType, TCoordRep >
{
public:

  /** Standard class typedefs. */
  typedef ReducedDimensionLinearInterpolateImageFunction   Self;
  typedef InterpolateImageFunction< TImageType, TCoordRep > Superclass;
  typedef SmartPointer< Self >                              Pointer;
  typedef SmartPointer< const Self >                        ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( ReducedDimensionLinearInterpolateImageFunction, InterpolateImageFunction );

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** OutputType typedef support. */
  typedef typename Superclass::OutputType OutputType;

  typedef typename Superclass::InputImageType  InputImageType;
  typedef typename InputImageType::SpacingType InputImageSpacingType;

  /** Dimension underlying input image. */
  itkStaticConstMacro( ImageDimension, unsigned int, Superclass::ImageDimension );

  /** Index typedef support. */
  typedef typename Superclass::IndexType IndexType;

  /** ContinuousIndex typedef support. */
  typedef typename Superclass::ContinuousIndexType ContinuousIndexType;
  typedef typename ContinuousIndexType::ValueType  ContinuousIndexValueType;

  /** PointType typedef support */
  typedef typename Superclass::PointType PointType;
    
  /** RealType typedef support. */
  typedef typename Superclass::RealType RealType;

  virtual OutputType EvaluateAtContinuousIndex( const ContinuousIndexType & index ) const
  {
      return this->EvaluateOptimized(Dispatch< ImageDimension >(), index);
  }
  /** Derivative typedef support */
  typedef CovariantVector< OutputType, itkGetStaticConstMacro( ImageDimension ) > CovariantVectorType;

  /** Method to compute the derivative. */
  CovariantVectorType EvaluateDerivativeAtContinuousIndex( const ContinuousIndexType & x ) const;

  /** Method to compute both the value and the derivative. */
  void EvaluateValueAndDerivativeAtContinuousIndex( const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const
  {
    return this->EvaluateValueAndDerivativeOptimized( Dispatch< ImageDimension >(), x, value, deriv );
  }

protected:

  ReducedDimensionLinearInterpolateImageFunction();
  virtual ~ReducedDimensionLinearInterpolateImageFunction() {}
  void PrintSelf( std::ostream & os, Indent indent ) const;

private:

  ReducedDimensionLinearInterpolateImageFunction( const Self & ); //purposely not implemented
  void operator=( const Self & );                                  //purposely not implemented

  /** Helper struct to select the correct dimension. */
  struct DispatchBase {};
  template< unsigned int >
  struct Dispatch : public DispatchBase {};
    
    OutputType EvaluateOptimized(const Dispatch< 3 > &, const ContinuousIndexType & index) const;

  OutputType EvaluateOptimized(const Dispatch< 4 > &, const ContinuousIndexType & index) const;
    
  inline OutputType EvaluateOptimized(const DispatchBase &, const ContinuousIndexType & index) const
  {
    return this->EvaluateUnoptimized(index);
  }
    
  virtual inline OutputType EvaluateUnoptimized( const ContinuousIndexType & index) const
  {
    itkExceptionMacro( << "ERROR: EvaluateAtContinuosIndex() in ReducedDimensionLinearInterpolateImageFunction"
                        << "is not implemented for this dimension ("
                        << ImageDimension << ")." );
  }


  /** Method to compute both the value and the derivative. 3D specialization. */
  void EvaluateValueAndDerivativeOptimized( const Dispatch< 3 > &, const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const;
    
  /** Method to compute both the value and the derivative. 4D specialization. */
  void EvaluateValueAndDerivativeOptimized( const Dispatch< 4 > &, const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const;

  /** Method to compute both the value and the derivative. Generic. */
  inline void EvaluateValueAndDerivativeOptimized( const DispatchBase &, const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const
  {
    return this->EvaluateValueAndDerivativeUnOptimized( x, value, deriv );
  }
    
  /** Method to compute both the value and the derivative. Generic. */
  inline void EvaluateValueAndDerivativeUnOptimized( const ContinuousIndexType & x, OutputType & value, CovariantVectorType & deriv ) const
  {
    itkExceptionMacro( << "ERROR: EvaluateValueAndDerivativeAtContinuousIndex() in ReducedDimensionLinearInterpolateImageFunction"
                          << "is not implemented for this dimension ("
                          << ImageDimension << ")." );
  }

};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkReducedDimensionLinearInterpolateImageFunction.hxx"
#endif

#endif
