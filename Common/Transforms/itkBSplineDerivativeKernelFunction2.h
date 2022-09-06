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
  Module:    $RCSfile: itkBSplineDerivativeKernelFunction2.h,v $
  Language:  C++
  Date:      $Date: 2008-06-25 11:00:19 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkBSplineDerivativeKernelFunction2_h
#define itkBSplineDerivativeKernelFunction2_h

#include "itkKernelFunctionBase2.h"
#include <vnl/vnl_math.h>

namespace itk
{

/** \class BSplineDerivativeKernelFunction2
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
class ITK_TEMPLATE_EXPORT BSplineDerivativeKernelFunction2 : public KernelFunctionBase2<double>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineDerivativeKernelFunction2);

  /** Standard class typedefs. */
  using Self = BSplineDerivativeKernelFunction2;
  using Superclass = KernelFunctionBase2<double>;
  using Pointer = SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineDerivativeKernelFunction2, KernelFunctionBase2);

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
   * static (whereas this `Evaluate` member function is virtual). */
  static void
  FastEvaluate(const double u, double * const weights)
  {
    return Self::Evaluate(Dispatch<VSplineOrder>(), u, weights);
  }


  /** Evaluate the function. */
  inline double
  Evaluate(const double & u) const override
  {
    return Self::FastEvaluate(u);
  }


  /** Evaluate the function. */
  inline void
  Evaluate(const double & u, double * weights) const override
  {
    return Self::FastEvaluate(u, weights);
  }


protected:
  BSplineDerivativeKernelFunction2() = default;
  ~BSplineDerivativeKernelFunction2() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "Spline Order: " << SplineOrder << std::endl;
  }


private:
  /** Structures to control overloaded versions of Evaluate */
  template <unsigned int>
  struct ITK_TEMPLATE_EXPORT Dispatch
  {};

  /** Zeroth order spline. */
  // Derivative not defined.

  /** First order spline */
  inline static double
  Evaluate(const Dispatch<1> &, const double u)
  {
    const double absValue = std::abs(u);

    if (absValue < 1.0)
    {
      return -vnl_math::sgn(u);
    }
    else if (absValue == 1.0)
    {
      return -vnl_math::sgn(u) / 2.0;
    }
    else
    {
      return 0.0;
    }
  }


  inline static void
  Evaluate(const Dispatch<1> &, const double u, double * weights)
  {
    // MS \todo: check
    const double absValue = std::abs(u);

    if (absValue < 1.0 && absValue > 0.0)
    {
      weights[0] = -1.0;
      weights[1] = 1.0;
    }
    else if (absValue == 1)
    {
      weights[0] = -0.5;
      weights[1] = 0.0;
    }
    else
    {
      weights[0] = 0.0;
      weights[1] = 0.5;
    }
  }


  /** Second order spline. */
  inline static double
  Evaluate(const Dispatch<2> &, const double u)
  {
    double absValue = std::abs(u);

    if (absValue < 0.5)
    {
      return -2.0 * u;
    }
    else if (absValue < 1.5)
    {
      return u - 1.5 * vnl_math::sgn(u);
    }
    else
    {
      return 0.0;
    }
  }


  inline static void
  Evaluate(const Dispatch<2> &, const double u, double * weights)
  {
    // MS \todo: check
    weights[0] = u - 1.5;
    weights[1] = -2.0 * u + 2.0;
    weights[2] = u - 0.5;
  }


  /**  Third order spline. */
  inline static double
  Evaluate(const Dispatch<3> &, const double u)
  {
    const double absValue = std::abs(u);
    const double sqrValue = u * u;

    if (absValue < 1.0)
    {
      if (u > 0.0)
      {
        const double dummy = std::abs(u + 0.5);
        return (6.0 * sqrValue - 2.0 * u - 6.0 * dummy + 3.0) / 4.0;
      }
      else
      {
        const double dummy = std::abs(u - 0.5);
        return -(6.0 * sqrValue + 2.0 * u - 6.0 * dummy + 3.0) / 4.0;
      }
    }
    else if (absValue < 2.0)
    {
      if (u > 0.0)
      {
        const double dummy = std::abs(u - 0.5);
        return (u - sqrValue + 3.0 * dummy - 2.5) / 2.0;
      }
      else
      {
        const double dummy = std::abs(u + 0.5);
        return (u + sqrValue - 3.0 * dummy + 2.5) / 2.0;
      }
    }
    else
    {
      return 0.0;
    }
  }


  inline static void
  Evaluate(const Dispatch<3> &, const double u, double * weights)
  {
    const double absValue = std::abs(u);
    const double sqrValue = u * u;

    weights[0] = 0.5 * sqrValue - 2.0 * absValue + 2.0;
    weights[1] = -1.5 * sqrValue + 5.0 * absValue - 3.5;
    weights[2] = 1.5 * sqrValue - 4.0 * absValue + 2.0;
    weights[3] = -0.5 * sqrValue + absValue - 0.5;
  }
};

} // end namespace itk

#endif
