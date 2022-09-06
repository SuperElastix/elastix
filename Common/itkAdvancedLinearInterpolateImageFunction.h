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
#ifndef itkAdvancedLinearInterpolateImageFunction_h
#define itkAdvancedLinearInterpolateImageFunction_h

#include "itkLinearInterpolateImageFunction.h"

namespace itk
{
/** \class AdvancedLinearInterpolateImageFunction
 * \brief Linearly interpolate an image at specified positions.
 *
 * AdvancedLinearInterpolateImageFunction linearly interpolates image intensity at
 * a non-integer pixel position. This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * This function works for N-dimensional images.
 *
 * This function works for images with scalar and vector pixel
 * types, and for images of type VectorImage.
 *
 * Unlike the LinearInterpolateImageFunction, which implements a constant
 * boundary condition, this class implements a mirroring boundary condition,
 * which mimics the BSplineInterpolateImageFunction.
 *
 * Edge cases, i.e. points exactly on the right most edge of the image,
 * need to be dealt with separately. In this implementation we subtract a
 * small number from the continuous index and interpolate at that position.
 * Alternatively, you would need to implement 7 different possibilities in
 * 3D, e.g.:
 *   x[0] is at end index           -> interpolate in x-y plane
 *   x[0] and x[1] are at end index -> interpolate along z line
 *   all are at end index           -> nearest neighbor interpolation
 * We opt to subtract a small number from x, which is computationally efficient,
 * gives cleaner code, and almost exactly the same interpolated value.
 *
 * \sa VectorAdvancedLinearInterpolateImageFunction
 *
 * \ingroup ImageFunctions ImageInterpolators
 * \ingroup ITKImageFunction
 *
 * \wiki
 * \wikiexample{ImageProcessing/LinearInterpolateImageFunction,Linearly interpolate a position in an image}
 * \endwiki
 */
template <class TInputImage, class TCoordRep = double>
class ITK_TEMPLATE_EXPORT AdvancedLinearInterpolateImageFunction
  : public LinearInterpolateImageFunction<TInputImage, TCoordRep>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedLinearInterpolateImageFunction);

  /** Standard class typedefs. */
  using Self = AdvancedLinearInterpolateImageFunction;
  using Superclass = LinearInterpolateImageFunction<TInputImage, TCoordRep>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(AdvancedLinearInterpolateImageFunction, LinearInterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType typedef support. */
  using typename Superclass::OutputType;

  /** InputImageType typedef support. */
  using typename Superclass::InputImageType;
  using InputImageSpacingType = typename InputImageType::SpacingType;

  /** InputPixelType typedef support. */
  using typename Superclass::InputPixelType;

  /** RealType typedef support. */
  using typename Superclass::RealType;

  /** Dimension underlying input image. */
  itkStaticConstMacro(ImageDimension, unsigned int, Superclass::ImageDimension);

  /** Index typedef support. */
  using typename Superclass::IndexType;

  /** ContinuousIndex typedef support. */
  using typename Superclass::ContinuousIndexType;
  using ContinuousIndexValueType = typename ContinuousIndexType::ValueType;

  /** Derivative typedef support */
  using CovariantVectorType = CovariantVector<OutputType, Self::ImageDimension>;

  /** Method to compute the derivative. */
  CovariantVectorType
  EvaluateDerivativeAtContinuousIndex(const ContinuousIndexType & x) const;

  /** Method to compute both the value and the derivative. */
  void
  EvaluateValueAndDerivativeAtContinuousIndex(const ContinuousIndexType & x,
                                              OutputType &                value,
                                              CovariantVectorType &       deriv) const
  {
    return this->EvaluateValueAndDerivativeOptimized(Dispatch<ImageDimension>(), x, value, deriv);
  }


protected:
  AdvancedLinearInterpolateImageFunction() = default;
  ~AdvancedLinearInterpolateImageFunction() override = default;

private:
  /** Helper struct to select the correct dimension. */
  struct DispatchBase
  {};
  template <unsigned int>
  struct Dispatch : public DispatchBase
  {};

  /** Method to compute both the value and the derivative. 2D specialization. */
  inline void
  EvaluateValueAndDerivativeOptimized(const Dispatch<2> &,
                                      const ContinuousIndexType & x,
                                      OutputType &                value,
                                      CovariantVectorType &       deriv) const;

  /** Method to compute both the value and the derivative. 3D specialization. */
  inline void
  EvaluateValueAndDerivativeOptimized(const Dispatch<3> &,
                                      const ContinuousIndexType & x,
                                      OutputType &                value,
                                      CovariantVectorType &       deriv) const;

  /** Method to compute both the value and the derivative. Generic. */
  inline void
  EvaluateValueAndDerivativeOptimized(const DispatchBase &,
                                      const ContinuousIndexType & x,
                                      OutputType &                value,
                                      CovariantVectorType &       deriv) const
  {
    return this->EvaluateValueAndDerivativeUnOptimized(x, value, deriv);
  }


  /** Method to compute both the value and the derivative. Generic. */
  inline void
  EvaluateValueAndDerivativeUnOptimized(const ContinuousIndexType & x,
                                        OutputType &                value,
                                        CovariantVectorType &       deriv) const
  {
    itkExceptionMacro(<< "ERROR: EvaluateValueAndDerivativeAtContinuousIndex() is not implemented for this dimension ("
                      << ImageDimension << ").");
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkAdvancedLinearInterpolateImageFunction.hxx"
#endif

#endif
