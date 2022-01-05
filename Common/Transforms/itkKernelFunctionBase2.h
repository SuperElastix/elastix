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
#ifndef itkKernelFunctionBase2_h
#define itkKernelFunctionBase2_h

#include "itkKernelFunctionBase.h"

namespace itk
{
/** \class KernelFunctionBase2
 * \brief Kernel used for density estimation and nonparameteric regression.
 *
 * This class encapsulates the smoothing kernel used for statistical density
 * estimation and nonparameteric regression. The basic idea of the kernel
 * approach is to weight observations by a smooth function (the kernel)
 * to created a smoothed approximation.
 *
 * Reference:
 * Silverman, B. W. (1986) Density Estimation. London: Chapman and Hall.
 *
 * \ingroup Functions
 * \ingroup ITKCommon
 */
template <typename TRealValueType = double>
class ITK_TEMPLATE_EXPORT KernelFunctionBase2 : public KernelFunctionBase<TRealValueType>
{
public:
  /** Standard class typedefs. */
  using Self = KernelFunctionBase2;
  using Superclass = KernelFunctionBase<TRealValueType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using RealType = TRealValueType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(KernelFunctionBase2, KernelFunctionBase);

  /** Evaluate the function. Subclasses must implement this. */
  TRealValueType
  Evaluate(const TRealValueType & u) const override = 0;

  /** Evaluate the function. Subclasses must implement this. */
  virtual void
  Evaluate(const TRealValueType & u, TRealValueType * weights) const = 0;

protected:
  KernelFunctionBase2() = default;
  ~KernelFunctionBase2() override = default;
};
} // end namespace itk

#endif // itkKernelFunctionBase2_h
