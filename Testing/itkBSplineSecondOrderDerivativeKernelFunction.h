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
#ifndef itkBSplineSecondOrderDerivativeKernelFunction_h
#define itkBSplineSecondOrderDerivativeKernelFunction_h

#include "itkKernelFunctionBase.h"
#include "itkBSplineKernelFunction.h"

namespace itk
{

/** \class BSplineSecondOrderDerivativeKernelFunction
 * \brief Derivative of a BSpline kernel used for density estimation and
 *  nonparameteric regression.
 *
 * This class encapsulates the derivative of a BSpline kernel for
 * density estimation or nonparameteric regression.
 * See documentation for KernelFunction for more details.
 *
 * This class is templated over the spline order.
 * \warning Evaluate is only implemented for spline order 1 to 4
 *
 * \sa KernelFunction
 *
 * \ingroup Functions
 */
template <unsigned int VSplineOrder = 3>
class ITK_TEMPLATE_EXPORT BSplineSecondOrderDerivativeKernelFunction : public KernelFunctionBase<double>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineSecondOrderDerivativeKernelFunction);

  /** Standard class typedefs. */
  using Self = BSplineSecondOrderDerivativeKernelFunction;
  using Superclass = KernelFunctionBase<double>;
  using Pointer = SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(BSplineSecondOrderDerivativeKernelFunction);

  /** Enum of for spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Evaluate the function. */
  inline double
  Evaluate(const double & u) const override
  {
    return (KernelType::FastEvaluate(u + 1.0) - 2.0 * KernelType::FastEvaluate(u) + KernelType::FastEvaluate(u - 1.0));
  }


protected:
  using KernelType = BSplineKernelFunction<Self::SplineOrder - 2>;

  BSplineSecondOrderDerivativeKernelFunction() = default;

  ~BSplineSecondOrderDerivativeKernelFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "Spline Order: " << SplineOrder << std::endl;
  }
};

} // end namespace itk

#endif
