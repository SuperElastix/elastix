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
#ifndef itkGPURecursiveGaussianImageFilterFactory_h
#define itkGPURecursiveGaussianImageFilterFactory_h

#include "itkGPUObjectFactoryBase.h"
#include "itkGPURecursiveGaussianImageFilter.h"

namespace itk
{
/** \class GPURecursiveGaussianImageFilterFactory2
 * \brief Object Factory implementation for GPURecursiveGaussianImageFilter
 *
 * \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands
 *
 * \note This work was funded by the Netherlands Organisation for
 * Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
 *
 */
template <typename TTypeListIn, typename TTypeListOut, typename NDimensions>
class ITK_TEMPLATE_EXPORT GPURecursiveGaussianImageFilterFactory2 : public GPUObjectFactoryBase<NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(GPURecursiveGaussianImageFilterFactory2);

  using Self = GPURecursiveGaussianImageFilterFactory2;
  using Superclass = GPUObjectFactoryBase<NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Return a descriptive string describing the factory. */
  const char *
  GetDescription() const override
  {
    return "A Factory for GPURecursiveGaussianImageFilter";
  }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPURecursiveGaussianImageFilterFactory2, GPUObjectFactoryBase);

  /** Register one factory of this type. */
  static void
  RegisterOneFactory();

  /** Operator() to register override. */
  template <typename TTypeIn, typename TTypeOut, unsigned int VImageDimension>
  void
  operator()()
  {
    // Image typedefs
    using InputImageType = Image<TTypeIn, VImageDimension>;
    using OutputImageType = Image<TTypeOut, VImageDimension>;
    using GPUInputImageType = GPUImage<TTypeIn, VImageDimension>;
    using GPUOutputImageType = GPUImage<TTypeOut, VImageDimension>;

    // Override default
    this->RegisterOverride(
      typeid(RecursiveGaussianImageFilter<InputImageType, OutputImageType>).name(),
      typeid(GPURecursiveGaussianImageFilter<InputImageType, OutputImageType>).name(),
      "GPU RecursiveGaussianImageFilter override default",
      true,
      CreateObjectFunction<GPURecursiveGaussianImageFilter<InputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is first template argument
    this->RegisterOverride(
      typeid(RecursiveGaussianImageFilter<GPUInputImageType, OutputImageType>).name(),
      typeid(GPURecursiveGaussianImageFilter<GPUInputImageType, OutputImageType>).name(),
      "GPU RecursiveGaussianImageFilter override GPUImage first",
      true,
      CreateObjectFunction<GPURecursiveGaussianImageFilter<GPUInputImageType, OutputImageType>>::New());

    // Override when itkGPUImage is second template argument
    this->RegisterOverride(
      typeid(RecursiveGaussianImageFilter<InputImageType, GPUOutputImageType>).name(),
      typeid(GPURecursiveGaussianImageFilter<InputImageType, GPUOutputImageType>).name(),
      "GPU RecursiveGaussianImageFilter override GPUImage second",
      true,
      CreateObjectFunction<GPURecursiveGaussianImageFilter<InputImageType, GPUOutputImageType>>::New());

    // Override when itkGPUImage is first and second template arguments
    this->RegisterOverride(
      typeid(RecursiveGaussianImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
      typeid(GPURecursiveGaussianImageFilter<GPUInputImageType, GPUOutputImageType>).name(),
      "GPU RecursiveGaussianImageFilter override GPUImage first and second",
      true,
      CreateObjectFunction<GPURecursiveGaussianImageFilter<GPUInputImageType, GPUOutputImageType>>::New());
  }


protected:
  GPURecursiveGaussianImageFilterFactory2();
  ~GPURecursiveGaussianImageFilterFactory2() override = default;

  /** Register methods for 1D. */
  void
  Register1D() override;

  /** Register methods for 2D. */
  void
  Register2D() override;

  /** Register methods for 3D. */
  void
  Register3D() override;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkGPURecursiveGaussianImageFilterFactory.hxx"
#endif

#endif // end #ifndef itkGPURecursiveGaussianImageFilterFactory_h
