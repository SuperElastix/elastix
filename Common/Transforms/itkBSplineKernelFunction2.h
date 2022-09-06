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
  Module:    $RCSfile: itkBSplineKernelFunction.h,v $
  Language:  C++
  Date:      $Date: 2006-03-18 20:13:35 $
  Version:   $Revision: 1.7 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkBSplineKernelFunction2_h
#define itkBSplineKernelFunction2_h

#include "itkKernelFunctionBase2.h"
#include <vnl/vnl_math.h>

namespace itk
{

/** \class BSplineKernelFunction2
 * \brief B-spline kernel used for density estimation and nonparameteric
 *  regression.
 *
 * This class encapsulates B-spline kernel for
 * density estimation or nonparametric regression.
 * See documentation for KernelFunction for more details.
 *
 * This class is templated over the spline order.
 * \warning Evaluate is only implemented for spline order 0 to 3
 *
 * \sa KernelFunction
 *
 * \ingroup Functions
 */
template <unsigned int VSplineOrder = 3>
class ITK_TEMPLATE_EXPORT BSplineKernelFunction2 : public KernelFunctionBase2<double>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(BSplineKernelFunction2);

  /** Standard class typedefs. */
  using Self = BSplineKernelFunction2;
  using Superclass = KernelFunctionBase2<double>;
  using Pointer = SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BSplineKernelFunction2, KernelFunctionBase2);

  /** Enum of for spline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Store weights for the entire support. */
  using WeightArrayType = FixedArray<double, VSplineOrder + 1>;

  /** Evaluate the function at one point. Faster than the corresponding public `Evaluate` member function, because it is
   * static (whereas this `Evaluate` member function is virtual). */
  static double
  FastEvaluate(const double u)
  {
    return Self::Evaluate(Dispatch<VSplineOrder>(), u);
  }


  /** Evaluate the function at the entire support. This is slightly faster, since no if's are needed. It is also faster
   * than the corresponding public `Evaluate` member function, because it is static (whereas this `Evaluate` member
   * function is virtual). */
  static void
  FastEvaluate(const double u, double * const weights)
  {
    Self::Evaluate(Dispatch<VSplineOrder>(), u, weights);
  }


  /** Evaluate the function at one point. */
  inline double
  Evaluate(const double & u) const override
  {
    return Self::FastEvaluate(u);
  }


  /** Evaluate the function at the entire support. This is slightly faster,
   * since no if's are needed.
   */
  inline void
  Evaluate(const double & u, double * weights) const override
  {
    Self::FastEvaluate(u, weights);
  }


protected:
  BSplineKernelFunction2() = default;
  ~BSplineKernelFunction2() override = default;

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

  /** *****************************************************
   * B-spline functions for one point.
   */

  /** Zeroth order spline. */
  inline static double
  Evaluate(const Dispatch<0> &, const double u)
  {
    const double absValue = std::abs(u);

    if (absValue < 0.5)
    {
      return 1.0;
    }
    else if (absValue == 0.5)
    {
      return 0.5;
    }
    else
    {
      return 0.0;
    }
  }


  /** First order spline */
  inline static double
  Evaluate(const Dispatch<1> &, const double u)
  {
    const double absValue = std::abs(u);

    if (absValue < 1.0)
    {
      return 1.0 - absValue;
    }
    else
    {
      return 0.0;
    }
  }


  /** Second order spline. */
  inline static double
  Evaluate(const Dispatch<2> &, const double u)
  {
    const double absValue = std::abs(u);

    if (absValue < 0.5)
    {
      return 0.75 - absValue * absValue;
    }
    else if (absValue < 1.5)
    {
      return (9.0 - 12.0 * absValue + 4.0 * absValue * absValue) / 8.0;
    }
    else
    {
      return 0.0;
    }
  }


  /** Third order spline. */
  inline static double
  Evaluate(const Dispatch<3> &, const double u)
  {
    const double absValue = std::abs(u);
    const double sqrValue = u * u;

    if (absValue < 1.0)
    {
      return (4.0 - 6.0 * sqrValue + 3.0 * sqrValue * absValue) / 6.0;
    }
    else if (absValue < 2.0)
    {
      return (8.0 - 12.0 * absValue + 6.0 * sqrValue - sqrValue * absValue) / 6.0;
    }
    else
    {
      return 0.0;
    }
  }


  /** *****************************************************
   * B-spline functions for all points in the support.
   */

  /** Zeroth order spline. */
  inline static void
  Evaluate(const Dispatch<0> &, const double u, double * weights)
  {
    const double absValue = std::abs(u);

    if (absValue < 0.5)
    {
      weights[0] = 1.0;
    }
    else if (absValue == 0.5)
    {
      weights[0] = 0.5;
    }
    else
    {
      weights[0] = 0.0;
    }
  }


  /** First order spline */
  inline static void
  Evaluate(const Dispatch<1> &, const double u, double * weights)
  {
    const double absValue = std::abs(u);

    weights[0] = 1.0 - absValue;
    weights[1] = absValue;
  }


  /** Second order spline. */
  inline static void
  Evaluate(const Dispatch<2> &, const double u, double * weights)
  {
    const double absValue = std::abs(u);
    const double sqrValue = u * u;

    weights[0] = (9.0 - 12.0 * absValue + 4.0 * sqrValue) / 8.0;
    weights[1] = -0.25 + 2.0 * absValue - sqrValue;
    weights[2] = (1.0 - 4.0 * absValue + 4.0 * sqrValue) / 8.0;
  }


  /**  Third order spline. */
  inline static void
  Evaluate(const Dispatch<3> &, const double u, double * weights)
  {
    const double absValue = std::abs(u);
    const double sqrValue = u * u;
    const double uuu = sqrValue * absValue;

    // Use (numerically) slightly less accurate multiplication with 1/6
    // instead of division by 6 to substantially improve speed.
    static const double onesixth = 1.0 / 6.0;
    weights[0] = (8.0 - 12.0 * absValue + 6.0 * sqrValue - uuu) * onesixth;
    weights[1] = (-5.0 + 21.0 * absValue - 15.0 * sqrValue + 3.0 * uuu) * onesixth;
    weights[2] = (4.0 - 12.0 * absValue + 12.0 * sqrValue - 3.0 * uuu) * onesixth;
    weights[3] = (-1.0 + 3.0 * absValue - 3.0 * sqrValue + uuu) * onesixth;
  }
};

} // end namespace itk

#endif
