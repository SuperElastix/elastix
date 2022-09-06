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
#ifndef itkBSplineSecondOrderDerivativeKernelFunction2_h
#define itkBSplineSecondOrderDerivativeKernelFunction2_h

#include "itkKernelFunctionBase.h"
#include <vnl/vnl_math.h>
#include <cmath> // For abs.

namespace itk
{

/** \class BSplineSecondOrderDerivativeKernelFunction2
 * \brief Derivative of a B-spline kernel used for density estimation and
 *  nonparametric regression.
 *
 * This class encapsulates the derivative of a B-spline kernel for
 * density estimation or nonparametric regression.
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
class ITK_TEMPLATE_EXPORT BSplineSecondOrderDerivativeKernelFunction2 : public KernelFunctionBase<double>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineSecondOrderDerivativeKernelFunction2);

  /** Standard class typedefs. */
  using Self = BSplineSecondOrderDerivativeKernelFunction2;
  using Superclass = KernelFunctionBase<double>;
  using Pointer = SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineSecondOrderDerivativeKernelFunction2, KernelFunctionBase);

  /** Enum of for spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Evaluate the function. Faster than the corresponding public `Evaluate` member function, because it is
   * static (whereas this `Evaluate` member function is virtual). */
  static double
  FastEvaluate(const double u)
  {
    return Self::Evaluate(Dispatch<VSplineOrder>(), u);
  }


  /** Evaluate the function. Faster than the corresponding public `Evaluate` member function, because it is
   * static, so it does not require the creation of a `BSplineSecondOrderDerivativeKernelFunction2` object. */
  static void
  FastEvaluate(const double u, double * const weights)
  {
    Self::Evaluate(Dispatch<VSplineOrder>(), u, weights);
  }


  /** Evaluate the function. */
  inline double
  Evaluate(const double & u) const override
  {
    return Self::FastEvaluate(u);
  }


  /** Evaluate the function. */
  inline void
  Evaluate(const double u, double * weights) const
  {
    Self::FastEvaluate(u, weights);
  }


protected:
  BSplineSecondOrderDerivativeKernelFunction2() = default;
  ~BSplineSecondOrderDerivativeKernelFunction2() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "Spline Order: " << SplineOrder << std::endl;
  }


private:
  /** Structures to control overloaded versions of Evaluate */
  struct DispatchBase
  {};
  template <unsigned int>
  struct ITK_TEMPLATE_EXPORT Dispatch : DispatchBase
  {};

  /** Zeroth order spline. */
  // Second order derivative not defined.

  /** First order spline */
  // Second order derivative not defined.

  /** Second order spline. */
  inline static double
  Evaluate(const Dispatch<2> &, const double u)
  {
    double absValue = std::abs(u);

    if (absValue < 0.5)
    {
      return -2.0;
    }
    else if (absValue == 0.5)
    {
      return -0.5;
    }
    else if (absValue < 1.5)
    {
      return 1.0;
    }
    else if (absValue == 1.5)
    {
      return 0.5;
    }
    else
    {
      return 0.0;
    }
  }


  inline static void
  Evaluate(const Dispatch<2> &, const double u, double * weights)
  {
    weights[0] = 1.0;
    weights[1] = -2.0;
    weights[2] = 1.0;
  }


  /**  Third order spline. */
  inline static double
  Evaluate(const Dispatch<3> &, const double u)
  {
    const double absValue = std::abs(u);

    if (absValue < 1.0)
    {
      return vnl_math::sgn0(u) * (3.0 * u) - 2.0;
    }
    else if (absValue < 2.0)
    {
      return -vnl_math::sgn(u) * u + 2.0;
    }
    else
    {
      return 0.0;
    }
  }


  inline static void
  Evaluate(const Dispatch<3> &, const double u, double * weights)
  {
    weights[0] = -u + 2.0;
    weights[1] = 3.0 * u - 5.0;
    weights[2] = -3.0 * u + 4.0;
    weights[3] = u - 1.0;
  }


  /** Unimplemented spline order */
  inline static double
  Evaluate(const DispatchBase &, const double)
  {
    itkGenericExceptionMacro("Evaluate not implemented for spline order " << SplineOrder);
  }


  inline static void
  Evaluate(const DispatchBase &, const double, double *)
  {
    itkGenericExceptionMacro("Evaluate not implemented for spline order " << SplineOrder);
  }
};

} // end namespace itk

#endif
